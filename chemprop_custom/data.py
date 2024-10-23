import warnings

from torch.utils.data import Dataset, DataLoader

from chemprop.data.collate import collate_batch, collate_multicomponent
from chemprop.data.datasets import MoleculeDataset, MulticomponentDataset, ReactionDataset
from chemprop.data.samplers import ClassBalanceSampler, SeededSampler


class CombinedDataset(Dataset):
    def __init__(self, mols_dataset: MoleculeDataset, descs: list[str]):
        self.mols_dataset = mols_dataset
        self.descs = descs

    def __len__(self):
        return len(self.mols_dataset)

    def __getitem__(self, idx):
        mol_data = self.mols_dataset[idx]
        text_data = self.descs[idx]  # Get the corresponding description
        return mol_data, text_data  # Return a tuple of molecular data and text data
    
def build_dataloader(
    dataset: MoleculeDataset | ReactionDataset | MulticomponentDataset,
    train_descs: list[str] | None = None,  # New parameter for text descriptions
    batch_size: int = 64,
    num_workers: int = 0,
    class_balance: bool = False,
    seed: int | None = None,
    shuffle: bool = True,
    **kwargs,
):
    """Return a DataLoader for MoleculeDataset or CombinedDataset with text descriptions.

    Parameters
    ----------
    dataset : MoleculeDataset | ReactionDataset | MulticomponentDataset
        The dataset containing the molecules or reactions to load.
    train_descs : list[str] | None, default=None
        List of text descriptions corresponding to the molecules in the dataset.
    batch_size : int, default=64
        the batch size to load.
    num_workers : int, default=0
        the number of workers used to build batches.
    class_balance : bool, default=False
        Whether to perform class balancing (i.e., use an equal number of positive and negative
        molecules). Class balance is only available for single task classification datasets. Set
        shuffle to True in order to get a random subset of the larger class.
    seed : int, default=None
        the random seed to use for shuffling (only used when `shuffle` is `True`).
    shuffle : bool, default=False
        whether to shuffle the data during sampling.
    """
    
    # If train_descs is provided, create a CombinedDataset
    if train_descs is not None:
        dataset = CombinedDataset(dataset, train_descs)

    if class_balance:
        sampler = ClassBalanceSampler(dataset.Y, seed, shuffle)
    elif shuffle and seed is not None:
        sampler = SeededSampler(len(dataset), seed)
    else:
        sampler = None

    if isinstance(dataset, MulticomponentDataset):
        collate_fn = collate_multicomponent
    else:
        collate_fn = collate_batch

    if len(dataset) % batch_size == 1:
        warnings.warn(
            f"Dropping last batch of size 1 to avoid issues with batch normalization "
            f"(dataset size = {len(dataset)}, batch_size = {batch_size})"
        )
        drop_last = True
    else:
        drop_last = False

    return DataLoader(
        dataset,
        batch_size,
        sampler is None and shuffle,
        sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=drop_last,
        **kwargs,
    )