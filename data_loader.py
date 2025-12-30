from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SheetInfo:
    name: str
    path: str  # path inside xlsx zip, e.g. "xl/worksheets/sheet1.xml"


_NS_MAIN = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
_NS_REL = {"r": "http://schemas.openxmlformats.org/package/2006/relationships"}
_NS_RID = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def _col_letters_to_index(col_letters: str) -> int:
    col = 0
    for ch in col_letters:
        if not ("A" <= ch <= "Z"):
            continue
        col = col * 26 + (ord(ch) - ord("A") + 1)
    return col - 1


def _parse_cell_ref(ref: str) -> Tuple[int, int]:
    """
    e.g. "C5" -> (row_idx=4, col_idx=2)
    """

    col_letters = []
    row_digits = []
    for ch in ref:
        if ch.isalpha():
            col_letters.append(ch.upper())
        elif ch.isdigit():
            row_digits.append(ch)
    if not col_letters or not row_digits:
        raise ValueError(f"Invalid cell reference: {ref!r}")
    col_idx = _col_letters_to_index("".join(col_letters))
    row_idx = int("".join(row_digits)) - 1
    return row_idx, col_idx


def _read_xlsx_shared_strings(z) -> List[str]:
    import xml.etree.ElementTree as ET

    try:
        xml_bytes = z.read("xl/sharedStrings.xml")
    except KeyError:
        return []

    root = ET.fromstring(xml_bytes)
    strings: List[str] = []
    for si in root.findall("s:si", _NS_MAIN):
        # <si> can include multiple <t> in rich text; join them.
        parts = []
        for t in si.findall(".//s:t", _NS_MAIN):
            if t.text is not None:
                parts.append(t.text)
        strings.append("".join(parts))
    return strings


def _read_xlsx_sheets(z) -> List[SheetInfo]:
    import xml.etree.ElementTree as ET

    wb_root = ET.fromstring(z.read("xl/workbook.xml"))
    rels_root = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))

    rid_to_target: Dict[str, str] = {}
    for rel in rels_root.findall("r:Relationship", _NS_REL):
        rid = rel.attrib.get("Id")
        target = rel.attrib.get("Target")
        if rid and target:
            rid_to_target[rid] = target

    sheets: List[SheetInfo] = []
    for sheet in wb_root.findall("s:sheets/s:sheet", _NS_MAIN):
        name = sheet.attrib.get("name") or ""
        rid = sheet.attrib.get(f"{{{_NS_RID}}}id")
        if not rid:
            continue
        target = rid_to_target.get(rid)
        if not target:
            continue
        sheets.append(SheetInfo(name=name, path=f"xl/{target}"))
    return sheets


def _read_xlsx_sheet_as_rows(z, sheet_path: str, shared_strings: Sequence[str]) -> List[List[object]]:
    import xml.etree.ElementTree as ET

    root = ET.fromstring(z.read(sheet_path))

    cells: Dict[Tuple[int, int], object] = {}
    max_row = -1
    max_col = -1

    for row in root.findall(".//s:sheetData/s:row", _NS_MAIN):
        for c in row.findall("s:c", _NS_MAIN):
            ref = c.attrib.get("r")
            if not ref:
                continue

            try:
                r_idx, c_idx = _parse_cell_ref(ref)
            except ValueError:
                continue

            t = c.attrib.get("t")

            v_elem = c.find("s:v", _NS_MAIN)
            if v_elem is None or v_elem.text is None:
                # handle inline strings
                if t == "inlineStr":
                    t_elem = c.find("s:is/s:t", _NS_MAIN)
                    if t_elem is not None and t_elem.text is not None:
                        value: object = t_elem.text
                        cells[(r_idx, c_idx)] = value
                        max_row = max(max_row, r_idx)
                        max_col = max(max_col, c_idx)
                continue

            raw = v_elem.text.strip()
            if raw == "":
                continue

            if t == "s":
                try:
                    value = shared_strings[int(raw)]
                except Exception:
                    value = raw
            elif t == "b":
                value = bool(int(raw))
            else:
                # numeric or general; keep int if possible
                try:
                    if any(ch in raw.lower() for ch in [".", "e"]):
                        value = float(raw)
                    else:
                        value = int(raw)
                except Exception:
                    value = raw

            cells[(r_idx, c_idx)] = value
            max_row = max(max_row, r_idx)
            max_col = max(max_col, c_idx)

    if max_row < 0 or max_col < 0:
        return []

    rows: List[List[object]] = [[None] * (max_col + 1) for _ in range(max_row + 1)]
    for (r_idx, c_idx), value in cells.items():
        if 0 <= r_idx < len(rows) and 0 <= c_idx < len(rows[r_idx]):
            rows[r_idx][c_idx] = value
    return rows


def _rows_to_dataframe_with_header(
    rows: List[List[object]],
    *,
    required_columns: Sequence[str],
) -> Optional[pd.DataFrame]:
    if not rows:
        return None

    required = [str(c) for c in required_columns]
    header_row_idx: Optional[int] = None
    header_map: Dict[str, int] = {}

    scan_limit = min(len(rows), 50)
    for i in range(scan_limit):
        row = rows[i]
        row_str = [str(v).strip() for v in row if v is not None]
        if all(col in row_str for col in required):
            header_row_idx = i
            # map column -> position
            for col in required:
                for j, v in enumerate(row):
                    if v is not None and str(v).strip() == col:
                        header_map[col] = j
                        break
            break

    if header_row_idx is None or len(header_map) != len(required):
        return None

    records: List[Dict[str, object]] = []
    for row in rows[header_row_idx + 1 :]:
        rec: Dict[str, object] = {}
        empty = True
        for col in required:
            j = header_map[col]
            val = row[j] if j < len(row) else None
            rec[col] = val
            if val is not None and str(val).strip() != "":
                empty = False
        if not empty:
            records.append(rec)

    if not records:
        return None

    df = pd.DataFrame.from_records(records)
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=list(required))
    return df.reset_index(drop=True)


def read_measurement_table(
    path: Union[str, Path],
    *,
    sheet: Optional[str] = None,
    required_columns: Sequence[str] = ("样机编号", "重量", "实际温度", "芯片温度", "信号"),
) -> pd.DataFrame:
    """
    读取测量数据表，支持：
    - CSV（utf-8-sig）
    - XLSX（不依赖 openpyxl，解析 xlsx 的 XML 内容；支持多 sheet）

    sheet:
    - None 或 "all": 读取所有包含 required_columns 的工作表并合并
    - 其他字符串：按 sheet 名称或 1-based 序号选择单个工作表（例如 "Sheet1" 或 "1"）
    """

    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path, encoding="utf-8-sig")
        return df

    if suffix != ".xlsx":
        raise ValueError(f"Unsupported file type: {path} (expected .csv or .xlsx)")

    import zipfile

    sheet_sel = (sheet or "").strip()
    if sheet_sel.lower() == "all":
        sheet_sel = ""

    with zipfile.ZipFile(path) as z:
        shared_strings = _read_xlsx_shared_strings(z)
        sheets = _read_xlsx_sheets(z)
        if not sheets:
            raise ValueError(f"No sheets found in xlsx: {path}")

        selected: List[SheetInfo] = []
        if sheet_sel:
            # by 1-based index
            if sheet_sel.isdigit():
                idx = int(sheet_sel)
                if idx < 1 or idx > len(sheets):
                    raise ValueError(f"sheet index out of range: {sheet_sel} (1..{len(sheets)})")
                selected = [sheets[idx - 1]]
            else:
                matches = [s for s in sheets if s.name == sheet_sel]
                if not matches:
                    raise ValueError(f"sheet not found: {sheet_sel!r}. Available: {[s.name for s in sheets]}")
                selected = matches[:1]
        else:
            selected = sheets

        frames: List[pd.DataFrame] = []
        for s in selected:
            rows = _read_xlsx_sheet_as_rows(z, s.path, shared_strings)
            df_sheet = _rows_to_dataframe_with_header(rows, required_columns=required_columns)
            if df_sheet is None or df_sheet.empty:
                continue
            df_sheet = df_sheet.copy()
            df_sheet["sheet"] = s.name
            frames.append(df_sheet)

    if not frames:
        raise ValueError(
            f"No valid sheets found in {path} containing columns {list(required_columns)}. "
            "Try specifying --sheet."
        )

    df_all = pd.concat(frames, ignore_index=True)
    # type cleanup
    df_all["样机编号"] = df_all["样机编号"].astype(int)
    df_all["重量"] = df_all["重量"].astype(float)
    df_all["实际温度"] = df_all["实际温度"].astype(float)
    df_all["芯片温度"] = df_all["芯片温度"].astype(float)
    df_all["信号"] = df_all["信号"].astype(float)
    return df_all

