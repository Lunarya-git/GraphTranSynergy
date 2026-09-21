# preprocess_data.py
# ─────────────────────────────────────────────────────────────────────────────
# Builds the PyTorch Geometric .pt files for new_labels_0_10 dataset.
# Must be run ONCE before training. Safe to re-run (skips if files exist).
# ─────────────────────────────────────────────────────────────────────────────

import os
import csv
import numpy as np
import pandas as pd
import networkx as nx
from rdkit import Chem
from utils_test import TestbedDataset

DATAFILE = 'new_labels_0_10'
CELLFILE = './data/new_cell_features_954.csv'
CSV_FILE = f'./data/{DATAFILE}.csv'

drug1_pt = f'data/processed/{DATAFILE}_drug1.pt'
drug2_pt = f'data/processed/{DATAFILE}_drug2.pt'

if os.path.isfile(drug1_pt) and os.path.isfile(drug2_pt):
    print(f'Processed files already exist — skipping preprocessing.')
    print(f'  {drug1_pt}')
    print(f'  {drug2_pt}')
    exit(0)

print('Starting data preprocessing...')
print(f'  CSV   : {CSV_FILE}')
print(f'  Cells : {CELLFILE}')

# ── Helper functions (copied from creat_data_DC.py) ──────────────────────────
def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception(f"input {x} not in allowable set {allowable_set}")
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom):
    return np.array(
        one_of_k_encoding_unk(atom.GetSymbol(),
            ['C','N','O','S','F','Si','P','Cl','Br','Mg','Na','Ca','Fe','As',
             'Al','I','B','V','K','Tl','Yb','Sb','Sn','Ag','Pd','Co','Se',
             'Ti','Zn','H','Li','Ge','Cu','Au','Ni','Cd','In','Mn','Zr','Cr',
             'Pt','Hg','Pb','Unknown']) +
        one_of_k_encoding(atom.GetDegree(), [0,1,2,3,4,5,6,7,8,9,10]) +
        one_of_k_encoding_unk(atom.GetTotalNumHs(), [0,1,2,3,4,5,6,7,8,9,10]) +
        one_of_k_encoding_unk(atom.GetImplicitValence(), [0,1,2,3,4,5,6,7,8,9,10]) +
        [atom.GetIsAromatic()]
    )

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))
    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = [[e1, e2] for e1, e2 in g.edges]
    return c_size, features, edge_index

# ── Load cell features ────────────────────────────────────────────────────────
print('Loading cell features...')
cell_features = []
with open(CELLFILE) as csvfile:
    for row in csv.reader(csvfile):
        cell_features.append(row)
cell_features = np.array(cell_features)
print(f'  Cell features shape: {cell_features.shape}')

# ── Build SMILES graph dict ───────────────────────────────────────────────────
print('Building SMILES graphs...')
df_smiles = pd.read_csv('./data/smiles.csv')
smile_graph = {}
for smile in set(df_smiles['smile'].tolist()):
    smile_graph[smile] = smile_to_graph(smile)
print(f'  {len(smile_graph)} unique SMILES processed')

# ── Load main dataset ─────────────────────────────────────────────────────────
print(f'Loading {CSV_FILE}...')
df = pd.read_csv(CSV_FILE)
drug1 = np.asarray(df['drug1'].tolist())
drug2 = np.asarray(df['drug2'].tolist())
cell  = np.asarray(df['cell'].tolist())
label = np.asarray(df['label'].tolist())
print(f'  Dataset size: {len(label)} samples')

# ── Create processed .pt files ────────────────────────────────────────────────
os.makedirs('data/processed', exist_ok=True)
print('Building drug1 graph dataset...')
TestbedDataset(root='data', dataset=DATAFILE + '_drug1',
               xd=drug1, xt=cell, xt_featrue=cell_features, y=label,
               smile_graph=smile_graph)

print('Building drug2 graph dataset...')
TestbedDataset(root='data', dataset=DATAFILE + '_drug2',
               xd=drug2, xt=cell, xt_featrue=cell_features, y=label,
               smile_graph=smile_graph)

print('Preprocessing complete!')
print(f'  Saved: {drug1_pt}')
print(f'  Saved: {drug2_pt}')
