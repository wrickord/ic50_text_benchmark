# Standard library imports
import os
import random
from pathlib import Path
from collections import defaultdict

# Third-party imports
import numpy as np
import pandas as pd
from lightning import pytorch as pl
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from rdkit import Chem
from chemprop import data, featurizers, nn
from chemprop.models import MPNN

# Local imports

# CUDA
print(f'CUDA available: {torch.cuda.is_available()}')

# Consants
CUR_DIR = os.path.dirname(os.path.realpath('__file__'))


class IC50_Model():
    def __init__(self, 
                 input_path, 
                 device_ids,
                 num_workers, 
                 smiles_column, 
                 scaffold_smiles_column,
                 text_column,
                 target_columns,
                 text_encoder_model_name,
                 verbose=False):
        self.input_path = input_path
        self.device_ids = device_ids
        self.num_workers = num_workers
        self.smiles_column = smiles_column
        self.scaffold_smiles_column = scaffold_smiles_column
        self.text_column = text_column
        self.target_columns = target_columns
        self.text_encoder_model_name = text_encoder_model_name
        self.verbose = verbose

    def __call__(self):
        train_loader, val_loader, test_loader = self.split_data()
        model = self.get_model()
        self.train(model, train_loader, val_loader)
        results = self.test(model, test_loader)

        return results
    
    def inspect_data(X):
        print('Type:', X.type())
        print('Shape:', X.shape)
        print('Requires Grad:', X.requires_grad)
        print('Numerical Range: [{:.2f}, {:.2f}]'.format(X.min(), X.max()))
        print('Mean and Var: {:.2f}, {:.2f}'.format(X.mean(), X.var()))
    
    def inspect_dataloader(self, dataloader):
        print(f'Number of workers: {dataloader.num_workers}')
        print(f'Batch size: {dataloader.batch_size}')
        print(f'Pin memory: {dataloader.pin_memory}')
        
        dataset = dataloader.dataset
        print(f'Dataset length: {len(dataset)}')
        print(f'Dataset type: {type(dataset)}')

    def make_scaffold_split(self, mols, split_ratio=(0.8, 0.1, 0.1)):
        assert len(split_ratio) == 3, 'Split ratio must have 3 values'
        assert np.isclose(np.sum(split_ratio), 1), 'Ratios must sum to 1'
        
        # Step 1: Group by scaffold
        scaffold_groups = defaultdict(list)
        for idx, scaffold in enumerate(mols):
            scaffold_groups[scaffold].append(idx)
        
        # Step 2: Sort scaffolds by descending size
        scaffold_groups = sorted(
            scaffold_groups.items(), 
            key=lambda x: len(x[1]), 
            reverse=True
        )
        
        # Step 3: Determine the number of molecules for each set
        num_molecules = len(mols)
        num_test = int(np.floor(split_ratio[2] * num_molecules))
        num_val = int(np.floor(split_ratio[1] * num_molecules))

        # Step 4: Assign scaffolds to sets, adding larger scaffolds to training
        train_indices, test_indices, val_indices = [], [], []
        remaining_indices = []

        # First, handle large scaffolds
        for scaffold, indices in scaffold_groups:
            if len(indices) > num_test // 2:
                train_indices.extend(indices)  # Larger scaffolds go to training
            else:
                remaining_indices.append(indices)  # Keep smaller scaffolds

        # Shuffle the remaining scaffolds
        np.random.shuffle(remaining_indices)

        # Step 5: Distribute remaining scaffolds randomly into test, val, train 
        for indices in remaining_indices:
            if len(test_indices) + len(indices) <= num_test:
                test_indices.extend(indices)
            elif len(val_indices) + len(indices) <= num_val:
                val_indices.extend(indices)
            else:
                train_indices.extend(indices)

        # Shuffle the indices within each set for randomness
        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)
        np.random.shuffle(val_indices)

        return train_indices, test_indices, val_indices

    def make_assay_split(self, df, threshold):
        # Shuffle the list of ids for random distribution
        id_to_count = df['assay_id'].value_counts().to_dict()    
        ids = list(id_to_count.keys())
        random.shuffle(ids)
        
        # Initialize three lists
        list1 = []
        list2 = []
        list3 = []
        
        # Track the sum of counts in list2 and list3
        count_sum2 = 0
        count_sum3 = 0
        
        # Distribute IDs
        for id_ in ids:
            # Check if both list2 and list3 have sums less than the threshold
            if count_sum2 <= threshold or count_sum3 <= threshold:
                # Assign to the list with the smaller count sum first
                if count_sum2 <= count_sum3 and count_sum2 <= threshold:
                    list2.append(id_)
                    count_sum2 += id_to_count[id_]
                elif count_sum3 <= threshold:
                    list3.append(id_)
                    count_sum3 += id_to_count[id_]
            else:
                # Add remaining ids to list1 once both sums exceed threshold
                list1.append(id_)
        
        return list1, list2, list3
    
    def write_split(self, df, indices, type, index=None):
        pd.DataFrame({
            'indices': indices,
            'smiles': df.loc[indices, self.smiles_column].values,
            'scaffold_smiles': df.loc[
                indices, 
                self.scaffold_smiles_column
            ].values,
            'log_value': df.loc[indices, 'log_value'].values
        }).sort_values('indices').to_csv(
            f'{CUR_DIR}/data/splits/{type}_split.csv' if index is None else
            f'{CUR_DIR}/data/splits/{type}_split_{index}.csv',
            index=False
        )

    def split_data(self, split_sizes=(0.8, 0.1, 0.1), split_idx=1):
        # Load data
        df_input = pd.read_csv(self.input_path)
        smis = df_input.loc[:, self.smiles_column].values
        ys = df_input.loc[:, self.target_columns].values
        all_data = [
            data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smis, ys)
        ]
        mols = [d.mol for d in all_data]

        if split_idx == 0:
            # Split data
            train_indices, val_indices, test_indices = self.make_scaffold_split(
                mols, split_sizes
            )
            self.write_split(df_input, train_indices, 'train')
            self.write_split(df_input, val_indices, 'val')
            self.write_split(df_input, test_indices, 'test')
        else:
            # Load split info from CSV
            train_indices = pd.read_csv(
                f'{CUR_DIR}/data/splits/train_split_{split_idx}.csv'
            )['train_indices'].values
            train_indices = np.random.permutation(train_indices)
            val_indices = pd.read_csv(
                f'{CUR_DIR}/data/splits/val_split_{split_idx}.csv'
            )['val_indices'].values
            val_indices = np.random.permutation(val_indices)
            test_indices = pd.read_csv(
                f'{CUR_DIR}/data/splits/test_split_{split_idx}.csv'
            )['test_indices'].values
            test_indices = np.random.permutation(test_indices)
        
        train_mols, val_mols, test_mols = data.split_data_by_indices(
            all_data, train_indices, val_indices, test_indices
        )

        if self.text_column is not None:
            texts = df_input.loc[:, self.text_column].values 
            train_texts = list(texts[train_indices])
            val_texts = list(texts[val_indices])
            test_texts = list(texts[test_indices])

            # # Tokenize text
            # tokenizer = AutoTokenizer.from_pretrained(
            #     self.text_encoder_model_name
            # )
            # tokenizer.pad_token = tokenizer.eos_token

            # train_texts = [
            #     tokenizer(
            #         train_text, 
            #         padding=True
            #     ).input_ids for train_text in train_texts
            # ]
            # val_texts = [
            #     tokenizer(
            #         val_text, 
            #         padding=True
            #     ).input_ids for val_text in val_texts
            # ]
            # test_texts = [
            #     tokenizer(
            #         test_text, 
            #         padding=True
            #     ).input_ids for test_text in test_texts
            # ]

        # Get train, val, and test datasets for molecules
        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

        mols_train_dset = data.MoleculeDataset(
            train_mols, 
            featurizer,
            train_texts if self.text_column is not None else None
        )
        self.scaler = mols_train_dset.normalize_targets()
        
        mols_val_dset = data.MoleculeDataset(
            val_mols,
            featurizer,
            val_texts if self.text_column is not None else None
        )
        mols_val_dset.normalize_targets(self.scaler)

        mols_test_dset = data.MoleculeDataset(
            test_mols,
            featurizer,
            test_texts if self.text_column is not None else None
        )

        # Inspect mols_train_dset
        print('Mols Train Dataset')
        print('Number of datapoints:', len(mols_train_dset))
        print(mols_train_dset[0])
        breakpoint()

        train_loader = data.build_dataloader(mols_train_dset, shuffle=False)
        val_loader = data.build_dataloader(mols_val_dset, shuffle=False)
        test_loader = data.build_dataloader(mols_test_dset, shuffle=False)

        return train_loader, val_loader, test_loader
    
    def get_model(self):
        # Message passing and aggregation
        mp = nn.BondMessagePassing()
        agg = nn.MeanAggregation()

        # Feed-forward network
        output_transform = nn.UnscaleTransform.from_standard_scaler(self.scaler)
        ffn = nn.RegressionFFN(output_transform=output_transform)

        # Batch normalization
        batch_norm = False

        # Get metrics–only the first metric is used for training and early 
        # stopping
        metric_list = [nn.metrics.RMSEMetric(), nn.metrics.MAEMetric()] 

        # Create encoders
        model = MPNN(
            mp, 
            agg, 
            ffn, 
            self.text_encoder_model_name, 
            batch_norm, 
            metric_list
        )

        # Summary
        print(model) if self.verbose else None

        return model

    def train(self, model, train_loader, val_loader):
        self.trainer = pl.Trainer(
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=True,
            accelerator='auto',
            devices=self.device_ids,
            max_epochs=50,
        )

        # Train moel
        self.trainer.fit(model, train_loader, val_loader)

    def test(self, model, test_loader):
        # Test model
        results = self.trainer.test(model, test_loader)

        return results


# Main function
if __name__ == '__main__':
    input_path = Path.cwd() / 'data' / 'chembl_regression.csv'
    device_ids = [5] # Set GPU device
    num_workers = 8
    smiles_column = 'smiles'
    scaffold_smiles_column = 'scaffold_smiles'
    text_column = 'description'
    target_columns = ['log_value']
    text_encoder_model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'

    model = IC50_Model(
        input_path, 
        device_ids,
        num_workers, 
        smiles_column,
        scaffold_smiles_column,
        text_column, 
        target_columns,
        text_encoder_model_name,
        verbose=True
    )
    results = model()