from abc import ABC, abstractmethod
from typing import Any, Callable, Generator
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import AddRandomWalkPE
from torch_geometric.data import InMemoryDataset, Data
from bcosgnn.data.data import brics_fragmentation_scheme, smiles_to_data
from bcosgnn.data.raw import get_smiles_and_targets
from tqdm import tqdm


class CatRandomWalkPE(AddRandomWalkPE):

    def forward(self, data: Data) -> Data:
        with_pe = super().forward(data)
        with_pe.x = torch.cat([with_pe.x, with_pe["random_walk_pe"]], dim=-1)
        return with_pe


pe_transform = CatRandomWalkPE(8)


class MolecularClassificationDataset(InMemoryDataset, ABC):

    def __init__(
        self,
        root: str | None = None,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(root, transform, pre_transform, **kwargs)
        self.load(self.processed_paths[0])

    @abstractmethod
    def provide_raw_data(
        self,
    ) -> Generator[tuple[str, int | list[int], dict[str, Any]], None, None]:
        raise NotImplementedError

    def num_classes(self) -> int:
        raise NotImplementedError

    def process(self) -> None:
        data_list = list()
        print("Creating data list...")
        for smiles, y, meta in tqdm(self.provide_raw_data()):
            data = smiles_to_data(
                smiles,
                fragmentation_scheme=brics_fragmentation_scheme,
            )
            if isinstance(y, int):
                y = [y]
                data.y = torch.tensor(y, dtype=torch.float32)
            elif isinstance(y, list):
                y_ = torch.zeros(self.num_classes(), dtype=torch.float32)
                y_.index_fill_(0, y, 1.0)
                data.y = y_
            for key in meta:
                data[key] = meta[key]
            data_list.append(pe_transform(data))
        self.save(data_list, self.processed_paths[0])


class SAureusDataset(MolecularClassificationDataset):

    def raw_file_names(self) -> list[str]:
        return ["s_aureus_curated_with_scores.csv"]

    def processed_file_names(self) -> list[str]:
        return ["s_aureus_curated_with_scores.pt"]

    def provide_raw_data(self) -> Generator[tuple[str, int], None, None]:
        smiles, y = get_smiles_and_targets(self.raw_paths[0], "is_susceptible")
        for s, t in zip(smiles, y):
            yield s, t, {}
