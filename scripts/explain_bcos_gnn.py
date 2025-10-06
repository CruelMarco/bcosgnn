import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import os.path as osp

from bcosgnn.models import BinaryClassifierGNN
from bcosgnn.data.splits import split_random
from bcosgnn.data.load_dataset import load_dataset, NamedDataset
from pathlib import Path


dataset_name = NamedDataset.DEBUG
# TODO make this a cli arg?
train_run_dir = Path("lightning_logs/version_22/")

if __name__ == "__main__":
    dataset = load_dataset(dataset_name)
    train_idx, val_idx, test_idx = split_random(dataset)
    test_loader = DataLoader(dataset[test_idx], batch_size=1, shuffle=False)
    model = BinaryClassifierGNN.load_from_checkpoint(
        list((train_run_dir / "checkpoints").glob("*.ckpt"))[0]
    )
    model.eval()
    explanations = dict()
    for data in tqdm(test_loader):
        explanation = model.explain(data)

        # TODO use attributes other than the molecule smiles as identifier/dict key?
        explanations[data.smiles[0]] = explanation
    torch.save(explanations, osp.join(train_run_dir, "explanations.pt"))
