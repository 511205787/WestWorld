#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
URDF -> MJCF (MuJoCo XML) conversion (mesh-free version):
- Automatically replace <mesh filename="..."/> with
  <box size="0.001 0.001 0.001"/>
- No STL/DAE or other mesh assets are required
- Dependency: pip install urdf2mjcf

Example usage:
    python urdf_to_mjcf_no_mesh.py --in OpenX_robot_xml --out OpenX_robot_mjcf
"""

import argparse
import sys
import tempfile
from pathlib import Path
import xml.etree.ElementTree as ET

try:
    from urdf2mjcf import run as urdf2mjcf_run
except Exception as e:
    sys.exit(
        "Missing dependency: 'urdf2mjcf'. Install it first:\n  pip install urdf2mjcf\n"
        f"Import error: {e}"
    )

def find_urdfs(folder: Path):
    return sorted([p for p in folder.rglob("*.urdf") if p.is_file()])

def replace_mesh_with_box(urdf_path: Path) -> Path:
    """
    Replace <geometry><mesh .../></geometry> with
    <geometry><box size="..."/></geometry>.
    Write the processed URDF to a temporary directory and return its path.
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # Replace every <mesh> node with a <box>.
    for mesh in root.findall(".//mesh"):
        # Rename <mesh .../> to <box> and clear its attributes.
        mesh.tag = "box"
        for k in list(mesh.attrib.keys()):
            del mesh.attrib[k]
        mesh.set("size", "0.001 0.001 0.001")  # Tiny placeholder box

    tmp_dir = Path(tempfile.mkdtemp(prefix="urdf_nomesh_"))
    out_urdf = tmp_dir / urdf_path.name
    tree.write(out_urdf, encoding="utf-8", xml_declaration=True)
    return out_urdf

def convert_one(urdf_path: Path, out_dir: Path) -> bool:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_xml = out_dir / (urdf_path.stem + ".xml")
    try:
        staged = replace_mesh_with_box(urdf_path)
        # Meshes have already been removed, so no assets need to be copied.
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
    ap = argparse.ArgumentParser(description="URDF->MJCF conversion (mesh-free, using placeholder boxes)")
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
