from rdkit import Chem
import torch
from torch_geometric.data import Data
import pandas as pd
from tqdm import tqdm



def smiles_to_graph(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
    except:
        return None
    if mol is None:
        return None

    # ======================
    # node features (CATEGORICAL ONLY)
    # ======================
    atom_features = []

    for atom in mol.GetAtoms():

        atom_type = atom.GetAtomicNum()  # OK (used as index)
        
        chirality_map = {
            Chem.rdchem.ChiralType.CHI_UNSPECIFIED: 0,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW: 1,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW: 2,
            Chem.rdchem.ChiralType.CHI_OTHER: 0
        }
        chirality = chirality_map.get(atom.GetChiralTag(), 0)

        atom_features.append([
            atom_type,
            chirality
        ])

    x = torch.tensor(atom_features, dtype=torch.long)

    # ======================
    # edge features (CATEGORICAL ONLY)
    # ======================
    edge_index = []
    edge_attr = []

    bond_type_map = {
        Chem.rdchem.BondType.SINGLE: 0,
        Chem.rdchem.BondType.DOUBLE: 1,
        Chem.rdchem.BondType.TRIPLE: 2,
        Chem.rdchem.BondType.AROMATIC: 3
    }

    stereo_map = {
        Chem.rdchem.BondStereo.STEREONONE: 0,
        Chem.rdchem.BondStereo.STEREOZ: 1,
        Chem.rdchem.BondStereo.STEREOE: 2
    }

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        bt = bond_type_map.get(bond.GetBondType(), 0)
        bd = stereo_map.get(bond.GetStereo(), 0)

        edge_index.append([i, j])
        edge_index.append([j, i])

        edge_attr.append([bt, bd])
        edge_attr.append([bt, bd])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

if __name__ == "__main__":
    df = pd.read_csv("data/combined_compounds.csv")
    failed = 0
    graphs = []

    for smiles in tqdm(df["smiles"]):
        g = smiles_to_graph(smiles)
        if g is not None:
            graphs.append(g)
        else:
            failed += 1

    print("total graphs:", len(graphs))
    print("failed:", failed)
    torch.save(graphs, "data/egfr_graphs.pt")
    print("saved to data/egfr_graphs.pt")