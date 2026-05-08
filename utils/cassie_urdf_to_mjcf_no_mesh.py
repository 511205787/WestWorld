#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
URDF -> MJCF (MuJoCo XML) conversion (robust version: fixes zero
mass/zero inertia and automatically adds collision elements)
Dependency: pip install urdf2mjcf

Example usage:
  python urdf_to_mjcf_robust.py --in OpenX_robot_xml --out OpenX_robot_mjcf
"""

import argparse
import sys
import tempfile
from pathlib import Path
import xml.etree.ElementTree as ET

# ---------- Tunable parameters ----------
PLACEHOLDER_BOX = "0.01 0.01 0.01"   # Placeholder geometry size (meters)
EPS_MASS        = 1e-6               # Small positive mass
EPS_INERTIA     = 1e-8               # Small positive inertia diagonal
# --------------------------------

try:
    from urdf2mjcf import run as urdf2mjcf_run
except Exception as e:
    sys.exit(
        "Missing dependency: 'urdf2mjcf'. Install it first:\n  pip install urdf2mjcf\n"
        f"Import error: {e}"
    )

NS = {"xacro": "http://ros.org/wiki/xacro"}  # Keep the xacro namespace if present

def find_urdfs(folder: Path):
    return sorted([p for p in folder.rglob("*.urdf") if p.is_file()])

def _ensure_inertial(link: ET.Element):
    inertial = link.find("inertial")
    if inertial is None:
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "origin", {"rpy":"0 0 0", "xyz":"0 0 0"})
        ET.SubElement(inertial, "mass", {"value": str(EPS_MASS)})
        ET.SubElement(inertial, "inertia", {
            "ixx":str(EPS_INERTIA),"ixy":"0","ixz":"0",
            "iyy":str(EPS_INERTIA),"iyz":"0","izz":str(EPS_INERTIA)
        })
        return

    # mass
    mass = inertial.find("mass")
    if mass is None or float(mass.get("value", "0") or "0") <= 0:
        if mass is None:
            mass = ET.SubElement(inertial, "mass")
        mass.set("value", str(EPS_MASS))

    # inertia
    inertia = inertial.find("inertia")
    if inertia is None:
        inertia = ET.SubElement(inertial, "inertia", {
            "ixx":str(EPS_INERTIA),"ixy":"0","ixz":"0",
            "iyy":str(EPS_INERTIA),"iyz":"0","izz":str(EPS_INERTIA)
        })
    else:
        vals = [float(inertia.get(k, "0") or "0") for k in ["ixx","ixy","ixz","iyy","iyz","izz"]]
        if all(v == 0 for v in vals):
            inertia.set("ixx", str(EPS_INERTIA))
            inertia.set("iyy", str(EPS_INERTIA))
            inertia.set("izz", str(EPS_INERTIA))
            inertia.set("ixy", "0"); inertia.set("ixz", "0"); inertia.set("iyz", "0")

    if inertial.find("origin") is None:
        ET.SubElement(inertial, "origin", {"rpy":"0 0 0", "xyz":"0 0 0"})

def _replace_mesh_in_geometry(geom: ET.Element):
    """Replace <mesh> under <geometry> with <box size=...>."""
    mesh = geom.find("mesh")
    if mesh is not None:
        # Rename the tag to box and clear mesh-specific attributes.
        mesh.tag = "box"
        for k in list(mesh.attrib.keys()):
            del mesh.attrib[k]
        mesh.set("size", PLACEHOLDER_BOX)

def _ensure_collision(link: ET.Element):
    """Add a placeholder box if <collision> is missing; replace collision meshes with boxes otherwise."""
    collisions = link.findall("collision")
    if not collisions:
        col = ET.SubElement(link, "collision")
        ET.SubElement(col, "origin", {"rpy":"0 0 0", "xyz":"0 0 0"})
        geom = ET.SubElement(col, "geometry")
        ET.SubElement(geom, "box", {"size": PLACEHOLDER_BOX})
    else:
        for col in collisions:
            geom = col.find("geometry")
            if geom is None:
                geom = ET.SubElement(col, "geometry")
                ET.SubElement(geom, "box", {"size": PLACEHOLDER_BOX})
            else:
                _replace_mesh_in_geometry(geom)

def _fix_visuals(link: ET.Element):
    """Replace visual meshes with boxes while keeping visual elements for MJCF visualization."""
    for vis in link.findall("visual"):
        geom = vis.find("geometry")
        if geom is None:
            geom = ET.SubElement(vis, "geometry")
            ET.SubElement(geom, "box", {"size": PLACEHOLDER_BOX})
        else:
            _replace_mesh_in_geometry(geom)

def stage_urdf(urdf_path: Path) -> Path:
    """Read and sanitize a URDF, then return the temporary output path."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # Process each link individually.
    for link in root.findall("link"):
        _ensure_inertial(link)
        _fix_visuals(link)
        _ensure_collision(link)

    # Defensively replace any remaining <geometry><mesh> entries with boxes.
    for geom in root.findall(".//geometry"):
        _replace_mesh_in_geometry(geom)

    tmp_dir = Path(tempfile.mkdtemp(prefix="urdf_nomesh_"))
    out_urdf = tmp_dir / urdf_path.name
    tree.write(out_urdf, encoding="utf-8", xml_declaration=True)
    return out_urdf

def convert_one(urdf_path: Path, out_dir: Path) -> bool:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_xml = out_dir / (urdf_path.stem + ".xml")
    try:
        staged = stage_urdf(urdf_path)
        urdf2mjcf_run(
            urdf_path=str(staged),
            mjcf_path=str(out_xml),
            copy_meshes=False,
        )
        print(f"OK: {urdf_path.name} -> {out_xml}")
        return True
    except Exception as e:
        print(f"FAIL: {urdf_path.name} ({e})")
        return False

def main():
    ap = argparse.ArgumentParser(description="URDF->MJCF conversion (fix zero mass/inertia and auto-add collision)")
    ap.add_argument("--in", dest="in_dir", type=Path, required=True, help="Input directory containing .urdf files")
    ap.add_argument("--out", dest="out_dir", type=Path, required=True, help="Output directory for MJCF (.xml) files")
    args = ap.parse_args()

    urdfs = find_urdfs(args.in_dir)
    if not urdfs:
        sys.exit(f"No .urdf files found under {args.in_dir}")

    print(f"Found {len(urdfs)} URDF(s). Converting to: {args.out_dir}")
    ok = 0
    for u in urdfs:
        ok += int(convert_one(u, args.out_dir))
    print(f"\nDone. {ok}/{len(urdfs)} converted successfully.")

if __name__ == "__main__":
    main()
