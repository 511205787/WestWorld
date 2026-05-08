#!/usr/bin/env python3
import argparse
import os
import zipfile
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape


NS = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _read_shared_strings(zf):
    strings = []
    try:
        with zf.open("xl/sharedStrings.xml") as f:
            tree = ET.parse(f)
        for si in tree.findall(".//main:si", NS):
            texts = [t.text or "" for t in si.findall(".//main:t", NS)]
            strings.append("".join(texts))
    except KeyError:
        return []
    return strings


def _sheet_name_to_path(zf):
    with zf.open("xl/workbook.xml") as f:
        wb_tree = ET.parse(f)
    sheets = wb_tree.findall(".//main:sheets/main:sheet", NS)
    name_to_rid = {}
    for s in sheets:
        name = s.attrib["name"]
        rid = s.attrib[
            "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
        ]
        name_to_rid[name] = rid

    with zf.open("xl/_rels/workbook.xml.rels") as f:
        rels_tree = ET.parse(f)
    rels = {
        r.attrib["Id"]: r.attrib["Target"]
        for r in rels_tree.findall(
            ".//{http://schemas.openxmlformats.org/package/2006/relationships}Relationship"
        )
    }
    name_to_path = {}
    for name, rid in name_to_rid.items():
        target = rels[rid]
        name_to_path[name] = "xl/" + target
    return name_to_path


def _load_sheet_rows(zf, sheet_path, shared_strings):
    with zf.open(sheet_path) as f:
        sheet_tree = ET.parse(f)
    rows = []
    for row in sheet_tree.findall(".//main:sheetData/main:row", NS):
        col_map = {}
        for c in row.findall("main:c", NS):
            cell_ref = c.attrib.get("r", "")
            if not cell_ref:
                continue
            col = ""
            for ch in cell_ref:
                if ch.isalpha():
                    col += ch
                else:
                    break
            col_idx = 0
            for ch in col:
                col_idx = col_idx * 26 + (ord(ch.upper()) - ord("A") + 1)
            col_idx -= 1

            cell_type = c.attrib.get("t")
            if cell_type == "s":
                v = c.find("main:v", NS)
                val = v.text or "" if v is not None else ""
                try:
                    idx = int(val)
                except Exception:
                    idx = -1
                text = shared_strings[idx] if 0 <= idx < len(shared_strings) else ""
            elif cell_type == "inlineStr":
                texts = [t.text or "" for t in c.findall(".//main:t", NS)]
                text = "".join(texts)
            else:
                v = c.find("main:v", NS)
                text = v.text or "" if v is not None else ""

            col_map[col_idx] = text
        if not col_map:
            rows.append([])
            continue
        max_col = max(col_map.keys())
        cells = [""] * (max_col + 1)
        for idx, val in col_map.items():
            cells[idx] = val
        rows.append(cells)
    return rows


def read_env_counts(path):
    with zipfile.ZipFile(path) as zf:
        shared_strings = _read_shared_strings(zf)
        name_to_path = _sheet_name_to_path(zf)
        if "env_counts" not in name_to_path:
            raise ValueError("env_counts sheet not found in the input workbook.")
        rows = _load_sheet_rows(zf, name_to_path["env_counts"], shared_strings)

    if not rows:
        return []
    max_len = max(len(r) for r in rows)
    rows = [r + [""] * (max_len - len(r)) for r in rows]
    header = rows[0]
    records = []
    for r in rows[1:]:
        if not any(str(x).strip() for x in r):
            continue
        rec = {header[i]: r[i] for i in range(len(header))}
        try:
            rec["episodes"] = int(float(rec.get("episodes", 0)))
        except Exception:
            rec["episodes"] = 0
        try:
            rec["transitions"] = int(float(rec.get("transitions", 0)))
        except Exception:
            rec["transitions"] = 0
        records.append(rec)
    return records


def _col_name(idx):
    name = ""
    while idx:
        idx, rem = divmod(idx - 1, 26)
        name = chr(65 + rem) + name
    return name


def _sheet_xml(rows):
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">',
        "<sheetData>",
    ]
    for r_idx, row in enumerate(rows, start=1):
        parts.append(f'<row r="{r_idx}">')
        for c_idx, value in enumerate(row, start=1):
            if value is None:
                continue
            cell_ref = f"{_col_name(c_idx)}{r_idx}"
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                parts.append(f'<c r="{cell_ref}"><v>{value}</v></c>')
            else:
                text = escape(str(value))
                parts.append(
                    f'<c r="{cell_ref}" t="inlineStr"><is><t>{text}</t></is></c>'
                )
        parts.append("</row>")
    parts.append("</sheetData></worksheet>")
    return "".join(parts)


def _workbook_xml(sheet_names):
    sheets_xml = []
    for idx, name in enumerate(sheet_names, start=1):
        sheets_xml.append(f'<sheet name="{escape(name)}" sheetId="{idx}" r:id="rId{idx}"/>')
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        "<sheets>"
        + "".join(sheets_xml)
        + "</sheets></workbook>"
    )


def _workbook_rels_xml(sheet_count):
    rels = []
    for idx in range(1, sheet_count + 1):
        rels.append(
            f'<Relationship Id="rId{idx}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{idx}.xml"/>'
        )
    rels.append(
        f'<Relationship Id="rId{sheet_count + 1}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(rels)
        + "</Relationships>"
    )


def _content_types_xml(sheet_count):
    overrides = [
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        '<Override PartName="/xl/styles.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>',
    ]
    for idx in range(1, sheet_count + 1):
        overrides.append(
            f'<Override PartName="/xl/worksheets/sheet{idx}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        + "".join(overrides)
        + "</Types>"
    )


def _styles_xml():
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="1"><font><sz val="11"/><color theme="1"/><name val="Calibri"/><family val="2"/></font></fonts>'
        '<fills count="2">'
        '<fill><patternFill patternType="none"/></fill>'
        '<fill><patternFill patternType="gray125"/></fill>'
        "</fills>"
        '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>'
        '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
        "</styleSheet>"
    )


def write_xlsx(path, sheets):
    sheet_names = [name for name, _ in sheets]
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", _content_types_xml(len(sheet_names)))
        zf.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
            'Target="xl/workbook.xml"/>'
            "</Relationships>",
        )
        zf.writestr("xl/workbook.xml", _workbook_xml(sheet_names))
        zf.writestr("xl/_rels/workbook.xml.rels", _workbook_rels_xml(len(sheet_names)))
        zf.writestr("xl/styles.xml", _styles_xml())
        for idx, (_, rows) in enumerate(sheets, start=1):
            zf.writestr(f"xl/worksheets/sheet{idx}.xml", _sheet_xml(rows))


def main():
    parser = argparse.ArgumentParser(description="Select environments and write Excel.")
    parser.add_argument(
        "--input",
        default="env_task_counts.xlsx",
        help="Input Excel with env_counts sheet.",
    )
    parser.add_argument(
        "--out",
        default="env_selection.xlsx",
        help="Output Excel path.",
    )
    parser.add_argument(
        "--min-episodes",
        type=int,
        default=5000,
        help="Minimum episodes for the environment pool.",
    )
    parser.add_argument(
        "--k",
        type=int,
        choices=[5, 10, 20, 40, 50, 60],
        default=None,
        help="If set, only output the K environments (top-K by transitions).",
    )
    args = parser.parse_args()

    records = read_env_counts(args.input)
    if not records:
        raise RuntimeError("No env records found in the input workbook.")

    pool = [r for r in records if r.get("episodes", 0) >= args.min_episodes]
    if not pool:
        raise RuntimeError("No environments meet the min-episodes threshold.")
    pool.sort(
        key=lambda r: (
            -int(r.get("transitions", 0)),
            -int(r.get("episodes", 0)),
            str(r.get("environment", "")),
        )
    )

    header = ["rank", "environment", "task_ids", "episodes", "transitions"]
    sheets = []
    if args.k is None:
        pool_rows = [header]
        for idx, r in enumerate(pool, start=1):
            pool_rows.append(
                [
                    idx,
                    r.get("environment", ""),
                    r.get("task_ids", ""),
                    int(r.get("episodes", 0)),
                    int(r.get("transitions", 0)),
                ]
            )
        sheets.append(("pool", pool_rows))

        for k in [5, 10, 20, 40]:
            if k > len(pool):
                raise RuntimeError(f"Requested K={k} but pool has {len(pool)} envs.")
            rows = [header]
            for idx, r in enumerate(pool[:k], start=1):
                rows.append(
                    [
                        idx,
                        r.get("environment", ""),
                        r.get("task_ids", ""),
                        int(r.get("episodes", 0)),
                        int(r.get("transitions", 0)),
                    ]
                )
            sheets.append((f"K{k}", rows))
    else:
        if args.k > len(pool):
            raise RuntimeError(f"Requested K={args.k} but pool has {len(pool)} envs.")
        rows = [header]
        for idx, r in enumerate(pool[: args.k], start=1):
            rows.append(
                [
                    idx,
                    r.get("environment", ""),
                    r.get("task_ids", ""),
                    int(r.get("episodes", 0)),
                    int(r.get("transitions", 0)),
                ]
            )
        sheets.append((f"K{args.k}", rows))

    sheets.append(("budgets", [["budget_episodes"], [1000], [2000], [5000]]))

    out_path = os.path.abspath(args.out)
    write_xlsx(out_path, sheets)
    print(f"[Done] wrote {out_path}")


if __name__ == "__main__":
    main()

'''
python scripts/1_select_envs.py --input env_task_counts.xlsx --out env_selection_K50.xlsx --min-episodes 1500 --k 50

python scripts/1_select_envs.py --input env_task_counts.xlsx --out env_selection_K60.xlsx --min-episodes 1000 --k 60
'''
