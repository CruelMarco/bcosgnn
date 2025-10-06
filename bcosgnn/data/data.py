from typing import Any, Protocol
import torch
from torch_geometric.data import Data
from rdkit import Chem
from ..fragmentation import Fragment, find_brics_fragments

BT = Chem.BondType


class FragmentationScheme(Protocol):

    def __call__(self, mol: Chem.Mol) -> list[Fragment]: ...


brics_fragmentation_scheme = lambda mol: find_brics_fragments(mol)[0]


class DataWithFragmentIndex(Data):

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key == "fragment_index":
            return self.num_fragments
        return super().__inc__(key, value, *args, **kwargs)


def smiles_to_data(
    smiles: str,
    known_atom_symbols: list[str] = ["H", "C", "N", "O", "S", "F", "Cl", "Br", "I"],
    known_bond_types: list[int] = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC],
    remove_hydrogens: bool = True,
    fragmentation_scheme: FragmentationScheme | None = None,
):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if remove_hydrogens:
        mol = Chem.RemoveHs(mol)

    if fragmentation_scheme is not None:
        data = DataWithFragmentIndex(smiles=smiles)
    else:
        data = Data(smiles=smiles)

    atom_type = torch.zeros(mol.GetNumAtoms(), len(known_atom_symbols) + 1)
    num_hydrogens = None
    if not remove_hydrogens:
        num_hydrogens = torch.zeros(mol.GetNumAtoms(), 5)
    is_aromatic = torch.zeros(mol.GetNumAtoms(), 1)
    is_in_ring = torch.zeros(mol.GetNumAtoms(), 1)
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        try:
            atom_type_index = known_atom_symbols.index(symbol)
        except ValueError:
            atom_type_index = -1
        atom_type[atom.GetIdx(), atom_type_index] = 1
        is_aromatic[atom.GetIdx()] = int(atom.GetIsAromatic())
        is_in_ring[atom.GetIdx()] = int(atom.IsInRing())
        if not remove_hydrogens:
            num_hydrogens[atom.GetIdx(), atom.GetTotalNumHs()] = 1
    x = [atom_type, is_aromatic, is_in_ring]
    if not remove_hydrogens:
        x.append(num_hydrogens)
    data.x = torch.cat(x, dim=-1).to(torch.float)

    bond_type = torch.zeros(mol.GetNumBonds() * 2, len(known_bond_types) + 1)
    is_bond_in_ring = torch.zeros(mol.GetNumBonds() * 2, 1)
    row, col = [], []
    for j, bond in enumerate(mol.GetBonds()):
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        try:
            bond_type_index = known_bond_types.index(bond.GetBondType())
        except ValueError:
            bond_type_index = -1
        for bond_index in (j * 2, j * 2 + 1):
            bond_type[bond_index, bond_type_index] = 1
            is_bond_in_ring[bond_index] = int(bond.IsInRing())
        row.extend([start, end])
        col.extend([end, start])

    data.edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_attr = [bond_type, is_bond_in_ring]
    data.edge_attr = torch.cat(edge_attr, dim=-1).to(torch.float)

    if fragmentation_scheme is not None:
        fragments = fragmentation_scheme(mol)
        fragment_index = torch.zeros(mol.GetNumAtoms(), dtype=torch.long)
        for fragment_id, fragment in enumerate(fragments):
            fragment_index[fragment.atom_index] = fragment_id
        data.fragment_index = fragment_index
        data.num_fragments = len(fragments)

    return data
