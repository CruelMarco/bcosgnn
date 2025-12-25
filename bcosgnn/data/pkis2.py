import polaris as po
from torch_geometric.data import InMemoryDataset
import pandas as pd
from tqdm import tqdm
import torch
from bcosgnn.data.data import smiles_to_data

CLASSES = [
    "CLS_EGFR",
    "CLS_KIT",
    "CLS_RET",
    "CLS_LOK",
    "CLS_SLK",
]


class AddLabelTensor:

    def __call__(self, data):
        data.y = torch.stack([data[cls] for cls in CLASSES], dim=1)
        return data


class Pkis2SubDataset(InMemoryDataset):
    expected_version: int = 1

    def __init__(
        self,
        root=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        log=True,
        force_reload=False,
    ):
        super().__init__(root, transform, pre_transform, pre_filter, log, force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [f"pkis2_subset_v{self.expected_version}.csv"]

    @property
    def processed_file_names(self):
        return [f"pkis2_subset_v{self.expected_version}.pt"]

    def download(self):
        dataset = po.load_dataset("polaris/drewry2017-pkis2-subset-v2")
        assert dataset.version == self.expected_version
        dataset.table.to_csv(self.raw_paths[0], index=False)

    def load_raw_data_frame(self) -> pd.DataFrame:
        return pd.read_csv(self.raw_paths[0])

    def process(self):
        data_frame = self.load_raw_data_frame()
        data_list = []
        for row in tqdm(data_frame.itertuples()):
            data = smiles_to_data(row.MOL_smiles)
            data["MOL_molhash_id"] = row.MOL_molhash_id
            for target_col in (
                "EGFR",
                "KIT",
                "RET",
                "LOK",
                "SLK",
                "CLS_EGFR",
                "CLS_KIT",
                "CLS_RET",
                "CLS_LOK",
                "CLS_SLK",
            ):
                y = getattr(row, target_col)
                data[target_col] = torch.tensor([y], dtype=torch.float32)
            data_list.append(data)
        self.save(data_list, self.processed_paths[0])
