"""Pretrained GIN (supervised masking; Hu et al. 2020, dgllife `gin_supervised_masking`) in plain PyTorch on GPU.

MapLight+GNN (arXiv 2310.00174) appends this model's mean-pooled 300-d node embedding to the MapLight fingerprints (via
molfeat's PretrainedDGLTransformer). DGL is not installable beside torch 2.14 on this aarch64 box, so this module ports
the dgllife pieces it uses, weight-compatible with the released checkpoint:
  * PretrainAtomFeaturizer: (atomic number - 1, chiral tag index); PretrainBondFeaturizer: (bond type, bond direction)
    for both directions of every bond, then one self-loop per atom with bond type 4 (mol_to_bigraph add_self_loop=True);
  * GIN(num_node_emb_list=[120, 3], num_edge_emb_list=[6, 3], 5 layers, 300-d, JK='last'): each layer sums
    h[src] + edge embedding over incoming edges, then MLP(300-600-300), BatchNorm (eval), ReLU except the last layer.
The port is validated end to end by reproducing MapLight+GNN's published TDC numbers (paper/tdc/run_maplight.py).
"""
from __future__ import annotations

import pathlib

import numpy as np
import torch
import torch.nn as nn

WEIGHTS = pathlib.Path(__file__).resolve().parents[2] / "data" / "tdc_features" / "gin_supervised_masking.pth"
URL = "https://data.dgl.ai/dgllife/pre_trained/gin_supervised_masking.pth"


def _types():
    from rdkit import Chem
    chiral = [Chem.rdchem.ChiralType.CHI_UNSPECIFIED, Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
              Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW, Chem.rdchem.ChiralType.CHI_OTHER]
    bonds = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    dirs = [Chem.rdchem.BondDir.NONE, Chem.rdchem.BondDir.ENDUPRIGHT, Chem.rdchem.BondDir.ENDDOWNRIGHT]
    return chiral, bonds, dirs


def featurize(mol) -> tuple[np.ndarray, np.ndarray]:
    """Node rows (atomic-number index, chirality index); edge rows (src, dst, bond type, bond direction)."""
    chiral, bonds, dirs = _types()
    nodes = np.array([[a.GetAtomicNum() - 1, min(chiral.index(a.GetChiralTag()), 2)] for a in mol.GetAtoms()], np.int64)
    edges = []
    for b in mol.GetBonds():
        bt = bonds.index(b.GetBondType()) if b.GetBondType() in bonds else 0
        bd = dirs.index(b.GetBondDir()) if b.GetBondDir() in dirs else 0
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        edges += [[u, v, bt, bd], [v, u, bt, bd]]
    edges += [[i, i, 4, 0] for i in range(mol.GetNumAtoms())]
    return nodes, np.array(edges, np.int64).reshape(-1, 4)


class GINLayer(nn.Module):
    def __init__(self, d: int = 300, last: bool = False):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Linear(2 * d, d))
        self.edge_embeddings = nn.ModuleList([nn.Embedding(6, d), nn.Embedding(3, d)])
        self.bn = nn.BatchNorm1d(d); self.last = last

    def forward(self, h, src, dst, bt, bd):
        m = h[src] + self.edge_embeddings[0](bt) + self.edge_embeddings[1](bd)
        h = self.bn(self.mlp(torch.zeros_like(h).index_add_(0, dst, m)))
        return h if self.last else torch.relu(h)


class GIN(nn.Module):
    def __init__(self, d: int = 300, layers: int = 5):
        super().__init__()
        self.node_embeddings = nn.ModuleList([nn.Embedding(120, d), nn.Embedding(3, d)])
        self.gnn_layers = nn.ModuleList([GINLayer(d, last=i == layers - 1) for i in range(layers)])

    def forward(self, x, src, dst, bt, bd):
        h = self.node_embeddings[0](x[:, 0]) + self.node_embeddings[1](x[:, 1])
        for layer in self.gnn_layers:
            h = layer(h, src, dst, bt, bd)
        return h


def load(device: str = "cuda") -> GIN:
    if not WEIGHTS.exists():
        import urllib.request
        WEIGHTS.parent.mkdir(parents=True, exist_ok=True); urllib.request.urlretrieve(URL, WEIGHTS)
    m = GIN(); m.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=False)["model_state_dict"])
    return m.to(device).eval()


@torch.no_grad()
def embed(mols, batch: int = 256, device: str = "cuda") -> np.ndarray:
    """(N, 300) mean-pooled final-layer node embeddings, float64 (MapLight passes dtype=float)."""
    model, out = load(device), []
    for i in range(0, len(mols), batch):
        graphs = [featurize(m) for m in mols[i:i + batch]]
        n = np.array([len(g[0]) for g in graphs]); off = np.r_[0, np.cumsum(n)[:-1]]
        x = torch.tensor(np.concatenate([g[0] for g in graphs]), device=device)
        e = torch.tensor(np.concatenate([g[1] + np.array([o, o, 0, 0]) for g, o in zip(graphs, off)]), device=device)
        h = model(x, e[:, 0], e[:, 1], e[:, 2], e[:, 3])
        gid = torch.tensor(np.repeat(np.arange(len(graphs)), n), device=device)
        pooled = torch.zeros(len(graphs), h.shape[1], device=device).index_add_(0, gid, h) / torch.tensor(n, device=device)[:, None]
        out.append(pooled.double().cpu().numpy())
    return np.concatenate(out)
