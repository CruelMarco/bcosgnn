import torch
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
import seaborn as sns


def visualize_fp_contribution_map(
    smiles: str,
    fp: torch.Tensor,
    contribution_map: torch.Tensor,
):
    assert fp.dim() == contribution_map.dim() == 1
    assert fp.size(0) == contribution_map.size(0)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    img = Draw.MolToImage(mol, size=(300, 300))
