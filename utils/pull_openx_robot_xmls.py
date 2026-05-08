import os
import shutil
import sys
import importlib
from typing import Optional

'''
description: scripts to pull robotics struture file (including urdf or mjcf)
For urdf file need to use "urdf_to_mjcf_no_mesh.py" to transform to mjcf file and put in the "./robotics_structure_xml" folder
'''
# 1) Install robot_descriptions if it is not already available
try:
    import robot_descriptions  # noqa: F401
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "robot_descriptions"])

# 2) Output directory
OUT_DIR = os.path.abspath("robot_xml")
os.makedirs(OUT_DIR, exist_ok=True)

# 3) Target model list
robots = [
    {"pkg": "fanuc_m710ic_description", "model": "M-710iC", "brand": "Fanuc", "format": "URDF", "license": "BSD-3-Clause"},
    {"pkg": "a1_mj_description",         "model": "A1",      "brand": "UNITREE Robotics", "format": "MJCF", "license": "(per upstream repo)"},
    {"pkg": "iiwa7_description",         "model": "iiwa 7",  "brand": "KUKA", "format": "URDF", "license": "MIT"},
    {"pkg": "panda_mj_description",      "model": "Panda",   "brand": "Franka Robotics", "format": "MJCF", "license": "Apache-2.0"},
    {"pkg": "sawyer_mj_description",     "model": "Sawyer",  "brand": "Rethink Robotics", "format": "MJCF", "license": "Apache-2.0"},
    {"pkg": "ur5_description",           "model": "UR5",     "brand": "Universal Robots", "format": "URDF", "license": "Apache-2.0"},
]

def get_desc_path(mod, preferred: str) -> Optional[str]:
    """Prefer the declared format path; fall back to the alternative if needed."""
    preferred = preferred.upper()
    alt = "URDF" if preferred == "MJCF" else "MJCF"
    for fmt in (preferred, alt):
        attr = f"{fmt}_PATH"
        if hasattr(mod, attr):
            p = getattr(mod, attr)
            if isinstance(p, str) and os.path.isfile(p):
                return p
    return None

def pull_and_copy(pkg_name: str, fmt: str) -> str:
    mod = importlib.import_module(f"robot_descriptions.{pkg_name}")
    src = get_desc_path(mod, fmt)
    if not src:
        raise RuntimeError(f"{pkg_name}: could not find {fmt}_PATH (or a fallback path)")
    # Preserve the original extension (for example .urdf or .xml)
    ext = os.path.splitext(src)[1]
    dst = os.path.join(OUT_DIR, f"{pkg_name}{ext}")
    shutil.copy2(src, dst)
    return dst

rows = []
for r in robots:
    dst_path = pull_and_copy(r["pkg"], r["format"])
    rows.append({
        "Dataset": "OpenX",
        "Robotics type": r["brand"],
        "Model": r["model"],
        "Package": r["pkg"],
        "Format": r["format"],
        "License": r["license"],
        "xml file": dst_path,
    })

# Optional: write a CSV manifest
try:
    import pandas as pd
    df = pd.DataFrame(rows, columns=["Dataset","Robotics type","Model","Package","Format","License","xml file"])
    df.to_csv(os.path.join(OUT_DIR, "robot_xml_manifest.csv"), index=False, encoding="utf-8")
    print(df)
    print(f"\nSaved to: {os.path.join(OUT_DIR, 'robot_xml_manifest.csv')}")
except Exception:
    for row in rows:
        print(row)
