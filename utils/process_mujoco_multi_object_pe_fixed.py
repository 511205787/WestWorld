#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
process_mujoco_multi_object_pe_fixed.py

Load a (possibly multi‑object) MuJoCo XML model and build positional
embeddings that combine **object ID** and **three traversal ranks**
(pre‑order, LCRS in‑order, LCRS post‑order).

Changes vs. earlier draft
-------------------------
* The local `pre` list is kept as the **baseline order** for each object.
* `inlcrs` and `postlcrs` are mapped back to that baseline, so the three
  rank lists differ (match single‑object logic).
* Added clear comments, error guards, and logs.
"""

import os
import sys
import xmltodict
import numpy as np
import torch
import torch.nn as nn

# ── user settings ──────────────────────────────────────────────────────────
XML_FILE = "walker_generic.xml"               # MuJoCo XML path
TRAVERSAL_TYPES = ["pre", "inlcrs", "postlcrs"]
D_MODEL = 128                                 # output feature dim
DROP = 0.0                                     # dropout prob in PE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ───────────────────────────────────────────────────────────────────────────

# --------------------------------------------------------------------------
# XML helpers
# --------------------------------------------------------------------------

def collect_joints(node):
    """Safely collect all joint names inside a <body> subtree."""
    if not isinstance(node, dict):  # guard against stray text nodes
        return []

    names = []
    j = node.get("joint")
    if j:
        joints = j if isinstance(j, list) else [j]
        names.extend([jj["@name"] for jj in joints if isinstance(jj, dict)])

    # recurse over child <body> only if dict
    kids = node.get("body") or []
    for child in (kids if isinstance(kids, list) else [kids]):
        if isinstance(child, dict):
            names.extend(collect_joints(child))
    return names


def get_roots(xml):
    try:
        roots = xml["mujoco"]["worldbody"]["body"]
    except KeyError as e:
        raise RuntimeError("<worldbody><body> missing in XML") from e
    return roots if isinstance(roots, list) else [roots]

# --------------------------------------------------------------------------
# helper: collect all joints that any actuator drives
# --------------------------------------------------------------------------
def get_controlled_joints(xml_dict):
    """
    Return a set of joint names that are referenced by *any* actuator
    element (<motor>, <position>, <velocity>, <torque>, …).

    If an actuator points to a <tendon> instead of a joint we try to
    resolve that tendon and add the joints inside it as well.
    """
    controlled = set()

    act_root = xml_dict["mujoco"].get("actuator", {})
    if not isinstance(act_root, dict):
        return controlled                     # empty

    # 1) actuators that reference @joint directly --------------------------
    for tag_name, items in act_root.items():
        if tag_name.startswith("@"):          # skip attributes
            continue
        items = items if isinstance(items, list) else [items]
        for it in items:
            if isinstance(it, dict) and "@joint" in it:
                controlled.add(it["@joint"])

    # 2) actuators that reference @tendon ---------------------------------
    tendons = xml_dict["mujoco"].get("tendon", {})
    if tendons:
        # build mapping tendon-name -> joints inside that tendon
        tendon_jmap = {}
        for t_tag, ts in tendons.items():
            if t_tag.startswith("@"):
                continue
            ts = ts if isinstance(ts, list) else [ts]
            for t in ts:
                name = t.get("@name")
                if not name:
                    continue
                jnts = []
                for elem in (t.get("joint") or []):
                    for ji in (elem if isinstance(elem, list) else [elem]):
                        if isinstance(ji, dict) and "@joint" in ji:
                            jnts.append(ji["@joint"])
                tendon_jmap[name] = jnts

        # walk actuators with @tendon
        for tag_name, items in act_root.items():
            if tag_name.startswith("@"):
                continue
            items = items if isinstance(items, list) else [items]
            for it in items:
                if isinstance(it, dict) and "@tendon" in it:
                    controlled.update(tendon_jmap.get(it["@tendon"], []))

    return controlled


# --------------------------------------------------------------------------
# object ordering (root-body → object-ID)
# --------------------------------------------------------------------------
def order_objects(xml_dict):
    roots = get_roots(xml_dict)

    controlled = get_controlled_joints(xml_dict)
    if not controlled:
        print("[Warn] no actuator-controlled joints found → "
              "using first <body> as main object.")
        main = 0
    else:
        # pick first root whose subtree owns a controlled joint
        main = 0
        for i, body in enumerate(roots):
            if any(j in controlled for j in collect_joints(body)):
                main = i
                break

    # distance sort (same as before)
    def pos(body):
        return np.fromstring(body.get("@pos", "0 0 0"), sep=" ")
    main_pos = pos(roots[main])
    others = sorted(
        [(i, np.linalg.norm(pos(b) - main_pos))
         for i, b in enumerate(roots) if i != main],
        key=lambda x: x[1]
    )

    ordered = [main] + [i for i, _ in others]
    return roots, {orig: new for new, orig in enumerate(ordered)}


# --------------------------------------------------------------------------
# flatten hierarchy (pre‑order) -> parents, obj_ids
# --------------------------------------------------------------------------

def flatten(xml_dict, obj_map):
    roots = get_roots(xml_dict)
    if not isinstance(roots, list):
        roots = [roots]
    parents, obj_ids = [], []

    def dfs(node, parent, obj_id):
        idx = len(parents)
        parents.append(parent)
        obj_ids.append(obj_id)
        kids = node.get("body") or []
        for k in (kids if isinstance(kids, list) else [kids]):
            dfs(k, idx, obj_id)
    for orig, root in enumerate(roots):
        dfs(root, -1, obj_map[orig])
    return parents, obj_ids


# --------------------------------------------------------------------------
# traversals per object
# --------------------------------------------------------------------------

def children_lists(parents):
    ch = [[] for _ in parents]
    for i, p in enumerate(parents):
        if p >= 0:
            ch[p].append(i)
    return ch


def lcrs(nary):
    out = [[] for _ in nary]
    for u, kids in enumerate(nary):
        if not kids: continue
        out[u].append(kids[0])
        for a, b in zip(kids, kids[1:]):
            out[a].append(b)
    return out


def compute_traversals(parents, obj_ids):
    N = len(parents)
    ch = children_lists(parents)

    # nodes grouped by object
    groups = {}
    for i, o in enumerate(obj_ids):
        groups.setdefault(o, []).append(i)

    result = {}
    for o, nodes in groups.items():
        # local adjacency restricted to this object
        loc = {u: [v for v in ch[u] if obj_ids[v] == o] for u in nodes}

        # local roots: parent is -1 or belongs to different object
        roots = [u for u in nodes if parents[u] < 0 or obj_ids[parents[u]] != o]

        # 1. pre‑order -------------------------------------------------------
        pre = []
        def dfs_pre(u):
            pre.append(u)
            for v in loc[u]:
                dfs_pre(v)
        for r in roots:
            dfs_pre(r)

        # 2/3. build binary adj once
        bin_adj = [loc.get(u, []) for u in range(N)]
        bt = lcrs(bin_adj)

        # in‑order
        ino = []
        def dfs_in(u):
            if bt[u]: dfs_in(bt[u][0])
            ino.append(u)
            if len(bt[u]) == 2: dfs_in(bt[u][1])
        for r in roots: dfs_in(r)

        # post‑order
        post = []
        def dfs_post(u):
            for v in bt[u]: dfs_post(v)
            post.append(u)
        for r in roots: dfs_post(r)

        # map back to pre list to get ranks
        pre_rank   = list(range(len(pre)))
        in_rank    = [ino.index(u)  for u in pre]
        post_rank  = [post.index(u) for u in pre]

        result[o] = {
            "nodes"  : pre,        # baseline order
            "pre"    : pre_rank,
            "inlcrs" : in_rank,
            "postlcrs": post_rank,
        }
    return result


# --------------------------------------------------------------------------
# Embedding module
# --------------------------------------------------------------------------

class ConcatPositionalEmbedding(nn.Module):
    def __init__(self, d_model, rows_per_table, dropout=0.):
        super().__init__()
        self.num = len(rows_per_table)
        base = d_model // self.num
        self.tabs = nn.ModuleList()
        for i, rows in enumerate(rows_per_table):
            dim = base + (d_model % self.num if i == self.num - 1 else 0)
            self.tabs.append(nn.Embedding(rows, dim))
        self.drop = nn.Dropout(dropout)
    def forward(self, inds):
        return self.drop(torch.cat([t(inds[i]) for i, t in enumerate(self.tabs)], dim=1))


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

if __name__ == "__main__":
    if not os.path.isfile(XML_FILE):
        sys.exit(f"XML file not found: {XML_FILE}")

    with open(XML_FILE, "r", encoding="utf-8") as f:
        xml = xmltodict.parse(f.read())

    roots, obj_map = order_objects(xml)
    parents, obj_ids = flatten(xml, obj_map)
    local = compute_traversals(parents, obj_ids)

    # diagnostics
    for o, tr in local.items():
        print(f"[Object {o}] nodes={tr['nodes']}")
        for t in TRAVERSAL_TYPES:
            print(f"  {t:<7}: {tr[t]}")

    # build embedding tables
    max_sub = max(len(tr["nodes"]) for tr in local.values())
    rows = [len(local), max_sub, max_sub, max_sub]  # obj / pre / in / post
    enc = ConcatPositionalEmbedding(D_MODEL, rows, DROP).to(DEVICE)

    # produce embeddings per object
    for o, tr in local.items():
        n = len(tr["nodes"])
        idx_tensors = [
            torch.full((n,), o, dtype=torch.long, device=DEVICE),                 # obj id
            torch.tensor(tr["pre"],     dtype=torch.long, device=DEVICE),
            torch.tensor(tr["inlcrs"],  dtype=torch.long, device=DEVICE),
            torch.tensor(tr["postlcrs"],dtype=torch.long, device=DEVICE),
        ]
        pe = enc(idx_tensors)
        print(f"\n[Output] object {o} PE shape {tuple(pe.shape)}")
        print(pe)
