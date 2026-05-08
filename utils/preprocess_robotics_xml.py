#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""utils/preprocess_robotics_xml.py

Scan a directory of MuJoCo MJCF (XML) files and generate a YAML summary
for each XML file (all stored in a single YAML output), including:
  • Number of objects
  • Global pre-order index for each body and traversal ranks for three
    traversals (pre / inlcrs / postlcrs)
  • Optional "relative information" matrices for each object
    (PPR, symmetric Laplacian, shortest-path distance)

Usage:
python utils/preprocess_robotics_xml.py \
    --xml_dir  robotics_structure_xml \
    --out_yaml robotics_structure_xml/robotics_structure_summary.yaml
"""

from __future__ import annotations
import argparse, os, sys, yaml, xmltodict
from pathlib import Path
from collections import OrderedDict
from typing import Dict, List

import numpy as np

# Number of decimal places (None means no rounding)
_ROUND = 5
# Damping factor for PPR
_PPR_DAMPING = 0.9

# ---------------------------------------------------------------------------
#  Reused from the existing script: object ordering / flattening / traversal ranks
# ---------------------------------------------------------------------------
from process_mujoco_multi_object_pe_fixed import (
    order_objects, flatten, compute_traversals
)

# ---------------------------------------------------------------------------
#  Utility: collect body names in flatten pre-order for readability
# ---------------------------------------------------------------------------

def collect_body_names(xml_dict) -> List[str]:
    names: List[str] = []

    def dfs(node):
        names.append(node.get("@name", f"unnamed_{len(names)}"))
        kids = node.get("body") or []
        if not isinstance(kids, list):
            kids = [kids]
        for ch in kids:
            dfs(ch)

    for root in order_objects(xml_dict)[0]:
        dfs(root)
    return names  # Length matches the number of nodes returned by flatten

# ---------------------------------------------------------------------------
#  Graph features: adjacency / symmetric Laplacian / shortest-path distance / PPR
# ---------------------------------------------------------------------------

def _adj_from_parents(parents: List[int]) -> np.ndarray:
    """Build an undirected adjacency matrix from a parent-pointer list (no self-loops)."""
    N = len(parents)
    A = np.zeros((N, N), dtype=np.float32)
    for i, p in enumerate(parents):
        if p >= 0:
            A[i, p] = 1.0
            A[p, i] = 1.0
    return A

def _sym_laplacian(A: np.ndarray) -> np.ndarray:
    deg = A.sum(1)
    L = np.diag(deg) - A
    d12 = np.where(deg > 0.0, deg ** -0.5, 0.0)
    return (d12[:, None] * L) * d12[None, :]

def _distance(A: np.ndarray) -> np.ndarray:
    """All-pairs shortest-path distance for an unweighted graph using BFS. Returns [N, N], normalized by N."""
    N = A.shape[0]
    D = np.full((N, N), np.inf, dtype=np.float32)
    for s in range(N):
        D[s, s] = 0.0
        q = [s]
        while q:
            v = q.pop(0)
            for u in np.where(A[v] > 0)[0]:
                if D[s, u] == np.inf:
                    D[s, u] = D[s, v] + 1.0
                    q.append(u)
    D[np.isinf(D)] = 0.0
    return D / max(N, 1)

def _ppr(A: np.ndarray, alpha: float = _PPR_DAMPING) -> np.ndarray:
    """Personalized PageRank. Returns [N, N], where row i is the PPR starting from node i."""
    N = A.shape[0]
    # Add self-loops before computing the random-walk transition matrix.
    A_hat = A + np.eye(N, dtype=np.float32)
    deg = A_hat.sum(1, keepdims=True)
    T = (A_hat / np.maximum(deg, 1e-8)).T  # Column-stochastic
    I = np.eye(N, dtype=np.float32)
    inv = np.linalg.inv(I - alpha * T)
    P = (1.0 - alpha) * inv
    return P.T

def _maybe_round(x: np.ndarray):
    if _ROUND is None:
        return x.tolist()
    return np.round(x, _ROUND).tolist()

# ---------------------------------------------------------------------------
#  Single XML -> OrderedDict (can be dumped directly to YAML)
# ---------------------------------------------------------------------------

def process_single_xml(xml_path: Path) -> OrderedDict:
    with open(xml_path, "r", encoding="utf-8") as f:
        xml_dict = xmltodict.parse(f.read())

    body_names = collect_body_names(xml_dict)
    roots, obj_map   = order_objects(xml_dict)
    parents, obj_ids = flatten(xml_dict, obj_map)
    local            = compute_traversals(parents, obj_ids)

    # Full-graph adjacency using global pre-order indices.
    A_full = _adj_from_parents(parents)

    summary: OrderedDict[str, object] = OrderedDict()
    summary["xml"]         = str(xml_path)
    summary["num_objects"] = len(local)
    summary["objects"]     = []

    for o, tr in local.items():
        nodes = tr["nodes"]                        # Node indices for this object in global pre-order
        A = A_full[np.ix_(nodes, nodes)]          # Induced subgraph

        # Three relation matrices
        PPR = _ppr(A, alpha=_PPR_DAMPING)         # [n,n]
        LAP = _sym_laplacian(A)                   # [n,n]
        DIS = _distance(A)                        # [n,n]

        obj_od: OrderedDict[str, object] = OrderedDict()
        obj_od["obj_id"]     = int(o)
        obj_od["num_nodes"]  = len(nodes)
        obj_od["nodes"]      = []
        for local_idx, node_idx in enumerate(nodes):
            obj_od["nodes"].append(OrderedDict(
                body     = body_names[node_idx],
                idx      = int(node_idx),
                pre      = int(tr["pre"][local_idx]),
                inlcrs   = int(tr["inlcrs"][local_idx]),
                postlcrs = int(tr["postlcrs"][local_idx]),
            ))
        # relative information matrices
        # obj_od["relations"] = OrderedDict(
        #     types   = ["ppr", "sym_lap", "dist"],
        #     damping = float(_PPR_DAMPING),
        #     ppr     = _maybe_round(PPR),
        #     sym_lap = _maybe_round(LAP),
        #     dist    = _maybe_round(DIS),
        # )
        summary["objects"].append(obj_od)
    return summary

# ---------------------------------------------------------------------------
#  Convert NumPy scalars to builtins to keep YAML serialization safe
# ---------------------------------------------------------------------------

def to_builtin(obj):
    import numpy as np
    if isinstance(obj, OrderedDict):
        return {k: to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, dict):
        return {k: to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj

# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml_dir",  type=str, default="robotics_structure_xml",
                        help="Directory containing .xml MJCF files")
    parser.add_argument("--out_yaml", type=str, default="./robotics_structure_xml/robotics_structure_summary.yaml",
                        help="Output YAML path")
    args = parser.parse_args()

    xml_dir = Path(args.xml_dir)
    if not xml_dir.is_dir():
        sys.exit(f"[Error] {xml_dir} is not a directory")

    xml_files = sorted(xml_dir.glob("*.xml"))
    if not xml_files:
        sys.exit(f"[Error] No .xml files found in {xml_dir}")

    print(f"Found {len(xml_files)} MJCF files, processing …")
    summary: Dict[str, object] = OrderedDict()
    for xml_path in xml_files:
        key = xml_path.stem  # file name without extension
        summary[key] = process_single_xml(xml_path)
        print(f"  ✓ {key:<20s} → {summary[key]['num_objects']} object(s)")

    with open(args.out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(to_builtin(summary), f, sort_keys=False)
    print(f"\nSaved summary to {args.out_yaml}")

if __name__ == "__main__":
    main()
