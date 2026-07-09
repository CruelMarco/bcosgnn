from enum import Enum
from .graphs import SAureusDataset
from .mutag import AddSmilesToMutagData
import numpy as np
from torch_geometric.datasets import TUDataset


class NamedDataset(Enum):
    DEBUG = "debug"
    SAUREUS = "s_aureus"
    MUTAG = "mutag"


def load_dataset(
    dataset_name: NamedDataset,
    root="data",
):
    if dataset_name == NamedDataset.DEBUG:
        dataset = SAureusDataset(root=root)
        return dataset[np.random.random_integers(0, len(dataset), 100)]
    if dataset_name == NamedDataset.SAUREUS:
        return SAureusDataset(root=root)
    elif dataset_name == NamedDataset.MUTAG:
        return TUDataset(
            root=root, name="Mutagenicity", transform=AddSmilesToMutagData()
        )
    raise ValueError(f"Unknown dataset: {dataset_name}")
