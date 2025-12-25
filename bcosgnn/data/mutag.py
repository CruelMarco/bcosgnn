import torch
from rdkit import Chem

tu_atom_types = {
    0: "C",
    1: "O",
    2: "Cl",
    3: "H",
    4: "N",
    5: "F",
    6: "Br",
    7: "S",
    8: "P",
    9: "I",
    10: "Na",
    11: "K",
    12: "Li",
    13: "Ca",
}


def mol_from_mutag_data(data):
    mol = Chem.RWMol()
    for atom in data.x:
        atom_num = torch.where(atom == 1)[0].item()
        atom = Chem.Atom(tu_atom_types[atom_num])
        mol.AddAtom(atom)
    for bond, bt_num in zip(data.edge_index.T, data.edge_attr):
        begin_idx, end_idx = bond.tolist()
        if begin_idx > end_idx:
            continue
        bt_num = torch.where(bt_num == 1)[0].item()
        if bt_num == 0:
            bt = Chem.BondType.SINGLE
        elif bt_num == 1:
            bt = Chem.BondType.DOUBLE
        elif bt_num == 2:
            bt = Chem.BondType.TRIPLE
        mol.AddBond(bond[0].item(), bond[1].item(), bt)
    err = Chem.SanitizeMol(mol, catchErrors=True)
    if err != 0:
        print(f"Sanitization error: {err}")
    return mol


class AddSmilesToMutagData:
    def __call__(self, data):
        data.smiles = Chem.MolToSmiles(mol_from_mutag_data(data))
        return data
