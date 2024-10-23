## Imports
# Standard library imports
import os
from tqdm import tqdm
import nest_asyncio
nest_asyncio.apply()

# Third-party imports
import sqlite3
import numpy as np
import pandas as pd
import torch
from torch import nn
import lmdeploy
from lmdeploy import GenerationConfig
from transformers import pipeline as transformers_pipeline
from transformers import AutoTokenizer, LlamaForQuestionAnswering, AutoModelForCausalLM
from rdkit import Chem
from rdkit.Chem import Descriptors, Scaffolds
from rdkit.Chem.Scaffolds import MurckoScaffold

# Local imports

# GPU
print('CUDA available: ', torch.cuda.is_available())
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE_COUNT = torch.cuda.device_count()
print('Device Count: ', DEVICE_COUNT)  # Number of visible GPUs

## Constants
# Directory
CUR_DIR = os.path.dirname(os.path.realpath('__file__'))

# Chembl data
CYP3A4_CHEMBL_ID = 'CHEMBL340'
CHEMBL_DB_PATH = '/data/rbg/users/vincentf/data_uncertainty/chembl_34/' + \
    'chembl_34/chembl_34_sqlite/chembl_34.db'
JSON_PATH = f'{CUR_DIR}/data/chembl_data.json'

# Model
MODEL_NAME = 'meta-llama/Meta-Llama-3.1-8B-Instruct'


def pull_chembl_data():
    # Connect to chembl databse
    conn = sqlite3.connect(CHEMBL_DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Value types to extract
    value_types = [
        'IC50', 'IC5', 'Log IC50', 'pIC50', 'log(1/IC50)', '-Log IC50(M)', 
        'Ratio IC50', 'IC50(app)'
    ]
    placeholders = ','.join(['?'] * len(value_types))

    # Query
    query = f'''
        SELECT 
            targets.pref_name AS target,
            targets.chembl_id AS target_id,
            assays.chembl_id AS assay_id, 
            compound_structures.canonical_smiles AS smiles,
            activities.standard_type AS type,
            activities.standard_relation AS relation,
            activities.standard_value AS value,
            activities.standard_units AS unit,
            activities.molregno AS molregno,
            assays.doc_id AS doc_id,
            docs.journal AS journal,
            docs.doi AS doi,
            assays.description AS description,
            assays.assay_type AS assay_type, 
            assays.assay_test_type AS assay_test_type, 
            assays.assay_organism AS assay_organism, 
            assays.assay_tissue AS assay_tissue, 
            assays.assay_cell_type AS assay_cell_type, 
            assays.confidence_score AS assay_confidence_score, 
            activities.data_validity_comment AS data_validity_comment
        FROM 
            activities
        JOIN 
            target_dictionary AS targets ON assays.tid = targets.tid
        JOIN 
            assays ON activities.assay_id = assays.assay_id
        JOIN
            compound_structures USING (molregno)
        JOIN 
            docs ON assays.doc_id = docs.doc_id
        WHERE 
            targets.chembl_id = ?
            AND targets.target_type = 'SINGLE PROTEIN'
            AND activities.standard_type IN ({placeholders})
            AND activities.standard_value != 0
            AND activities.standard_value IS NOT NULL
        ORDER BY 
            activities.assay_id, value DESC
    '''

    # Execute query
    cur.execute(query, (CYP3A4_CHEMBL_ID, *value_types))

    # Fetch rows from the database
    rows = cur.fetchall()

    # Save as dataframe
    data = [dict(row) for row in rows]
    df = pd.DataFrame(data)
    df.to_json(f'{CUR_DIR}/data/chembl_data.json', orient='records')

    # Example
    for key, value in dict(rows[0]).items():
        print(key, ': ', value)

def fix_chembl_data():
    # Load into dataframe
    df = pd.read_json(JSON_PATH)
    print(df.head())

    # Investigate number of entries with a document listed
    chembl_df = df[
        (df['doi'].notnull())
    ].reset_index(drop=True)
    print(
        'Number of entries with a document attached:', 
        len(chembl_df) / len(df)
    )

    # Journal information (suggesting similar criteria for acceptance)
    journals = chembl_df['journal'].dropna().unique()
    print(f'\nNumber of journals within supporting documents: {len(journals)}')
    print(f'Journal names: {journals}')

    # Get the molecular weight from a smiles string
    def calculate_mol_weight(smiles):
        mol = Chem.MolFromSmiles(smiles)
        return Descriptors.ExactMolWt(mol)

    # Known unit conversion factors
    unit_dict = {'uM': 1, 'mM': 1e3, 'nM': 1e-3}

    # Function to get unit conversion factor
    def get_conversion_factor(unit, mol_weight=None):
        if unit == 'ug ml-1':
            return 1e3 / mol_weight
        elif unit == 'mg/ml':
            return 1e6 / mol_weight
        else:
            return unit_dict.get(unit)

    # Function to convert units
    def convert_units(row):
        smiles, value, unit = row['smiles'], row['value'], row['unit']
        mol_weight = calculate_mol_weight(smiles)
        conversion_factor = get_conversion_factor(unit, mol_weight)
        if conversion_factor:
            row['value'] = value * conversion_factor
            row['unit'] = 'uM'
        
        return row

    # Allowed unit types
    allowed_units = ['uM', 'nM']
    invalid_dois = [
        '10.1016/j.ejmech.2007.10.034', '10.1016/j.bmcl.2012.08.044', 
        '10.1021/acsmedchemlett.8b00220', '10.1016/j.ejmech.2008.12.004',
        '10.1016/j.bmcl.2015.01.005'
    ]

    # Correct discovered incorrect unit type for doi
    chembl_df.loc[chembl_df['doi'] == '10.1021/jm049696n', 'unit'] = 'nM'
    chembl_df.loc[chembl_df['doi'] == '10.1021/jm900521k', 'value'] /= 1e6
    chembl_df.loc[chembl_df['doi'] == '10.1021/jm900521k', 'unit'] = 'uM'

    # Convert ic50 value to correct units
    chembl_df = chembl_df.apply(convert_units, axis=1)

    # Remove rows with data not of interest
    chembl_df = chembl_df[
        (chembl_df['type'] == 'IC50') &
        (chembl_df['value'].notna()) &
        (chembl_df['value'] != 0) &
        (chembl_df['unit'].notna()) &
        (chembl_df['unit'].isin(allowed_units)) &
        (chembl_df['relation'].notna()) &
        (chembl_df['relation'] == '=') &
        ~(chembl_df['doi'].isin(invalid_dois)) &
        (chembl_df['description'].notna())
    ].sort_values(by='value', ascending=False)

    # Add log value column
    chembl_df['log_value'] = chembl_df['value'].apply(np.log10)
    cols = chembl_df.columns.insert(7, 'log_value')[:-1]
    chembl_df = chembl_df.reindex(columns=cols)

    print(f'Total number of activities for target: {len(chembl_df)}')

    # Add scaffold smiles column
    def get_scaffold(smiles):
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(
            Scaffolds.MurckoScaffold.GetScaffoldForMol(mol)
        )
    chembl_df['scaffold_smiles'] = chembl_df['smiles'].apply(get_scaffold)

    # Move column to after smiles
    cols = chembl_df.columns.insert(4, 'scaffold_smiles')[:-1]
    chembl_df = chembl_df.reindex(columns=cols)

    # Save data for regression task
    chembl_df.to_csv(f'{CUR_DIR}/data/chembl_regression.csv')
    chembl_df.to_json(
        f'{CUR_DIR}/data/chembl_regression.json', 
        orient='records'
    )
    print(chembl_df.info())
    print(chembl_df.head())

    print('Unique targets: ', chembl_df['target'].unique())
    print('Unique target ids: ', chembl_df['target_id'].unique())
    print('Unique types: ', chembl_df['type'].unique())
    print('Length of data: ', len(chembl_df))
    print('Number of unique smiles: ', len(chembl_df['smiles'].unique()))
    print(
        'Number of unique scaffolds: ', 
        len(chembl_df['scaffold_smiles'].unique())
    )
    print('Number of unique values: ', len(chembl_df['value'].unique()))
    print('Number of unique dois: ', len(chembl_df['doi'].unique()))

def get_doi_text():
    qa_model = LlamaForQuestionAnswering.from_pretrained(MODEL_NAME)
    sum_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    if DEVICE_COUNT > 1:
        qa_model = nn.DataParallel(qa_model)
        sum_model = nn.DataParallel(sum_model)

    qa_pipe = transformers_pipeline(
        'question-answering', 
        model=qa_model.module if isinstance(
            qa_model, 
            nn.DataParallel
        ) else qa_model,
        tokenizer=tokenizer,
        device=[0,1,2,3]
    )

    summarization_pipe = transformers_pipeline(
        'summarization', 
        model=sum_model.module if isinstance(
            sum_model, 
            nn.DataParallel
        ) else sum_model,
        tokenizer=tokenizer,
        device=[4,5,6,7]
    )
    # pipe = lmdeploy.pipeline(
    #     MODEL_NAME, 
    #     gen_config=GenerationConfig(
    #         max_new_tokens=4096,
    #         top_p=0.8,
    #         top_k=5,
    #         temperature=0.5
    #     )
    # )

    # Load data
    df = pd.read_json(f'{CUR_DIR}/data/chembl_regression.json')
    print(df.head())

    num_llm_entries = 0
    df['text_summary'] = None
    for i, row in df.iterrows():
        compound = row['smiles']
        value = row['value']
        doi = row['doi']
        print('Value: ', value)
        print('DOI: ', doi)

        # Check if text has already been processed
        if df.iloc[i]['text_summary']:
            continue

        # Load text
        try:
            txt_dir = '/data/rbg/users/vincentf/data_uncertainty/c340_txt/'
            txt_file = f'{doi.replace("/", "_")}.txt'
            txt_path = txt_dir + txt_file
            with open(txt_path, 'r', encoding='utf-8') as file:
                text = file.read()
        except:
            print('No text available for DOI: ', doi, '\n')
            continue

        def tokenize_text(text):
            tokens = tokenizer.encode(text, return_tensors='pt')
            return tokens
        
        def chunk_tokens(tokens, chunk_size=3872):
            return [tokens[:, i:i + chunk_size] for i in range(
                0, 
                tokens.size(1), 
                chunk_size // 2 # So that no important information is split
            )]
        
        responses = []
        chunks = chunk_tokens(tokenize_text(text))
        for chunk in tqdm(chunks, desc='Processing text chunks'):
            decoded_chunk = tokenizer.decode(chunk[0], skip_special_tokens=True)
            questions = [
                'What are the experimental conditions of the assay (e.g.,' + \
                    'substrate concentration, probe type, incubation time,' + \
                    'buffer conditions)?',
                'What method was used to determine the IC50 value (e.g.,' + \
                    'type of assay, detection method)?',
                'What other relevant details are there about how the IC50' + \
                    'was measured?'
            ]
            context = f'''
                Given the CYP3A4 (Cytochrome P450 3A4) IC50 measurement that
                produced the following result: {value} uM (might be in
                different units) for the following compound: {compound}.

                Find the information in this text: {decoded_chunk}.
                Use the exact wording from the text to answer the questions.

                If no text is present related to the requested information,
                please return "No information found" only once, no other
                information.
            '''    

            for question in questions:
                response = qa_pipe(
                    context=context, 
                    question=question, 
                    max_length=100, 
                    min_length=25, 
                    do_sample=False
                )
                responses.append(str(response['answer'].replace('\n', ' ')))   

            print(responses)
            
            summary = summarization_pipe(
                ' '.join([r['answer'] for r in responses]),
                max_length=100,
                min_length=25,
                do_sample=False
            )

            print(summary)
            break
        
        break     

            # prompt = f'''               
            #     Given the CYP3A4 (Cytochrome P450 3A4) IC50 measurement that 
            #     produced the following result: {value} uM (might be in 
            #     different units) for the following compound: {compound}.

            #     I am looking for only the following information:
            #     1.  The experimental conditions of the assay (e.g., substrate 
            #         concentration, probe type, incubation time, buffer 
            #         conditions).
            #     2.  The method used to determine the IC50 value (e.g., type of 
            #         assay, detection method).
            #     3.  Any other relevant details about how the IC50 was measured.

            #     Find the information in this text: {decoded_chunk}.
            #     Use the exact wording from the text to answer the questions.

            #     If no text is present related to the requested information, 
            #     please return "No information found" only once, no other 
            #     information.
            # '''
            

            # response = pipe([prompt])
            # responses.append(response[0].text)

        # prompt = f'''
        #     Given the CYP3A4 (Cytochrome P450 3A4) IC50 measurement that 
        #     produced the following result: {value} uM (might be in 
        #     different units).     

        #     Please condense the following text down to one statement regarding
        #     the following:
        #     1.  The experimental conditions of the assay (e.g., substrate 
        #             concentration, probe type, incubation time, buffer 
        #             conditions).
        #     2.  The method used to determine the IC50 value (e.g., type of 
        #         assay, detection method).
        #     3.  Any other relevant details about how the IC50 was measured.

        #     Here is the text: {' '.join(responses)}.

        #     Output only the response and not an introduction or conclusion.
        #     Remember, the response should only be in relation to the IC50
        #     measurement for CYP3A4 (Cytochrome P450 3A4).
        # '''
        # response = pipe([prompt])
        # df.iloc[
        #     i, 
        #     df.columns.get_loc('text_summary')
        # ] = str(response[0].text).replace('\n', ' ')

        # num_llm_entries += 1
        # if num_llm_entries % 10 == 0:
        #     print(f'Number of entries processed: {num_llm_entries}')

        #     # Save data
        #     df.to_csv(f'{CUR_DIR}/data/regression_with_texts.csv')
        #     df.to_json(
        #         f'{CUR_DIR}/data/regression_with_texts.json', 
        #         orient='records'
        #     )

    # Save data
    df.to_csv(f'{CUR_DIR}/data/regression_with_texts.csv')
    df.to_json(f'{CUR_DIR}/data/regression_with_texts.json', orient='records')


if __name__ == '__main__':
    # pull_chembl_data()
    # fix_chembl_data()
    get_doi_text()
