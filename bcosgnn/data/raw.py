import pandas as pd


def get_smiles_and_targets(
    csv_file,
    target_column,
    smiles_column="Smiles",
):
    raw = pd.read_csv(csv_file)
    smiles = raw[smiles_column].values
    y = raw[target_column].values.astype(int)
    return smiles, y
