"""TDC ADMET Group data access and frozen-trunk embedding cache.

Protocol is TDC's own (mims-harvard/TDC): `admet_group.get(name)` gives the official train_val/test split,
`get_train_valid_split(seed=s)` the official per-seed validation split, `evaluate_many` the leaderboard metric.
The trunk (ChemBERTa, pretrained on 77M molecules, frozen) is only ever used to embed SMILES; every arm sees
the same cached embeddings, so arms differ exactly by the bottleneck and nothing else.
"""
from __future__ import annotations

import pathlib

import torch
from tdc.benchmark_group import admet_group
from tdc.metadata import bm_metric_names

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "tdc"
CACHE = DATA / "emb"
TRUNK = "DeepChem/ChemBERTa-77M-MTR"
METRICS = bm_metric_names["admet_group"]
# TDC reports these metrics "higher is better"; the rest (mae, rmse) are lower-better.
HIGHER_IS_BETTER = {"roc-auc", "pr-auc", "spearman"}
CLASSIFICATION = {"roc-auc", "pr-auc"}


def group() -> admet_group:
    return admet_group(path=str(DATA))


def is_classification(endpoint: str) -> bool:
    return METRICS[endpoint] in CLASSIFICATION


def higher_is_better(endpoint: str) -> bool:
    return METRICS[endpoint] in HIGHER_IS_BETTER


def _embed(smiles: list[str], device: str = "cuda", batch: int = 256) -> torch.Tensor:
    """Mean-pooled last hidden state of the frozen trunk, one row per SMILES."""
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TRUNK)
    model = AutoModel.from_pretrained(TRUNK).to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(smiles), batch):
            enc = tok(smiles[i:i + batch], return_tensors="pt", padding=True, truncation=True, max_length=512)
            enc = {k: v.to(device) for k, v in enc.items()}
            h = model(**enc).last_hidden_state                       # (B, T, 384)
            mask = enc["attention_mask"].unsqueeze(-1).to(h.dtype)   # (B, T, 1)
            out.append(((h * mask).sum(1) / mask.sum(1).clamp(min=1)).float().cpu())
    return torch.cat(out)


def embeddings(endpoint: str, device: str = "cuda") -> dict[str, torch.Tensor]:
    """{smiles: vector} for every molecule of an endpoint, computed once and cached on disk."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{endpoint}.pt"
    if path.exists():
        blob = torch.load(path, weights_only=True)
        return dict(zip(blob["smiles"], blob["vecs"]))
    bench = group().get(endpoint)
    smiles = sorted(set(bench["train_val"].Drug) | set(bench["test"].Drug))
    vecs = _embed(smiles, device)
    torch.save({"smiles": smiles, "vecs": vecs}, path)
    return dict(zip(smiles, vecs))


def endpoints() -> list[str]:
    return list(group().dataset_names)


def mol_of(smi: str):
    """RDKit mol for a TDC SMILES. RDKit >= 2025 rejects a few official SMILES (e.g. solubility_aqsoldb's hypervalent
    `CC1=CC=C[NH+2]([O-])[CH-]1`); those are parsed unsanitised and sanitised without the valence check, so every
    official molecule keeps a prediction (TDC scores all test rows)."""
    from rdkit import Chem
    m = Chem.MolFromSmiles(smi)
    if m is None:
        m = Chem.MolFromSmiles(smi, sanitize=False)
        m.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(m, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES, catchErrors=True)
    return m
