"""Load and harmonise the raw NTD tables used by AI 2 (and AI 3).

Raw files stay untouched in ``data/raw``. Each loader returns one tidy table
covering every year on disk, with the same column names in every year:

* ``load_fuel_energy()``  NTD Fuel and Energy, one row per report_year x agency x mode x TOS
* ``load_vehicle_inventory()``  Revenue Vehicle Inventory, one row per fleet record x mode/TOS
* ``load_service()``  TS2.1 service time series (VOMS, VRM, VRH, UPT), one row per agency x mode x TOS x year

``build_interim()`` writes all three to ``data/interim/ntd/*.parquet``. Every
Parquet file carries the SHA-256 of the raw files it came from in its schema
metadata (key ``source_files``), so any result can be traced back and rebuilt.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim" / "ntd"

FUEL_DIR = RAW / "NTD_Fuel_energey"
RVI_DIR = RAW / "NTD Revenue Vehicle Inventory"
SERVICE_DIR = RAW / "NTD_service_data"

BUS_MODES = ("MB", "RB", "CB")  # bus, bus rapid transit, commuter bus

# Fuel keys used everywhere, and the unit NTD reports each quantity in.
FUEL_UNITS = {
    "diesel": "gal",
    "gasoline": "gal",
    "lpg": "gal",        # gallons equivalent
    "cng": "gal",        # gallons equivalent (GGE vs DGE: see docs/A2_phase4_0.md)
    "biodiesel": "gal",
    "hydrogen": "kg",    # reported separately only from 2022
    "other": "gal",
    "elec_prop": "kwh",  # overhead wire / third rail
    "elec_batt": "kwh",  # battery charging
}
QTY_COLS = [f"{f}_{u}" for f, u in FUEL_UNITS.items()]
MILES_FUELS = ["diesel", "gasoline", "lpg", "cng", "hydrogen", "other", "elec_prop", "elec_batt", "dual_fuel"]

# ---------------------------------------------------------------- helpers


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_ntd_id(s: pd.Series) -> pd.Series:
    """NTD IDs come as ints (1), floats (1.0) or strings ('00001'): make them 5-char strings."""
    num = pd.to_numeric(s, errors="coerce")
    out = num.astype("Int64").astype("string").str.zfill(5)
    return out.where(num.notna(), pd.NA)


def to_num(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        s = s.astype(str).str.replace(",", "", regex=False).str.strip().replace({"": None, "nan": None, "None": None})
    return pd.to_numeric(s, errors="coerce")


def _fuel_key(label: str) -> str | None:
    """Map a raw fuel label (Excel header or CSV prefix) to a FUEL_UNITS / MILES_FUELS key."""
    l = re.sub(r"\(.*?\)", "", str(label)).strip().lower().replace("-", "").replace("_", " ")
    l = re.sub(r"\s+", " ", l)
    table = [
        ("liquefied petroleum gas", "lpg"),
        ("compressed natural gas", "cng"),
        ("bio diesel", "biodiesel"), ("biodiesel", "biodiesel"),
        ("electric propulsion", "elec_prop"),
        ("electric battery", "elec_batt"),
        ("dual fuel", "dual_fuel"),
        ("other fuel", "other"),
        ("hydrogen", "hydrogen"),
        ("gasoline", "gasoline"),
        ("diesel", "diesel"),
    ]
    for k, v in table:
        if l.startswith(k):
            return v
    return None


def _write(df: pd.DataFrame, path: Path, sources: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    meta = dict(table.schema.metadata or {})
    meta[b"source_files"] = json.dumps(sources, indent=0).encode()
    pq.write_table(table.replace_schema_metadata(meta), path)


def source_hashes(path: Path) -> dict:
    """Read the raw-file hashes stored in a Parquet file written by this module."""
    md = pq.read_schema(path).metadata or {}
    return json.loads(md.get(b"source_files", b"{}"))


# ---------------------------------------------------------------- Fuel and Energy


def fuel_energy_files() -> dict[int, Path]:
    """Folder name = report year. Each folder holds one workbook (2015-2021) or two CSVs (2022+)."""
    out = {}
    for d in sorted(p for p in FUEL_DIR.iterdir() if p.is_dir() and p.name.isdigit()):
        files = [p for p in d.iterdir() if p.suffix.lower() in (".xlsx", ".xlsm", ".zip", ".csv")]
        csv_mode = [p for p in files if p.name.startswith("8ehq-7his")]  # per-mode table
        out[int(d.name)] = csv_mode[0] if csv_mode else files[0]
    return out


def _read_fuel_excel(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.lower().endswith((".xlsx", ".xlsm")))
            src = io.BytesIO(z.read(name))
    else:
        src = path
    raw = pd.read_excel(src, sheet_name="Fuel and Energy", engine="openpyxl", dtype=object)

    ident = {
        "Agency": "agency", "Name": "agency", "City": "city", "State": "state",
        "Legacy NTD ID": "legacy_ntd_id", "NTD ID": "ntd_id", "Organization Type": "organization_type",
        "Reporter Type": "reporter_type", "Primary UZA Population": "uza_population",
        "Agency VOMS": "agency_voms", "Mode": "mode", "TOS": "tos", "Mode VOMS": "mode_voms",
    }
    out = pd.DataFrame(index=raw.index)
    section, last = None, None
    for c in raw.columns:
        cs = str(c).strip()
        if cs in ident:
            out[ident[cs]] = raw[c]
            continue
        if cs.startswith("Fuel/Energy Used"):
            section = "qty"; continue
        if cs.startswith("Miles Traveled"):
            section = "miles"; continue
        if cs.startswith("Miles per Gallon"):
            section = "mpg"; continue
        if cs.startswith("Any data questionable"):
            out["any_questionable"] = raw[c]; section = None; continue
        if section is None:
            continue
        if cs.endswith("Questionable"):
            if last is not None:
                out[f"{last}_flag"] = raw[c]
            continue
        key = _fuel_key(cs)
        if key is None:
            continue
        name = {"qty": f"{key}_{FUEL_UNITS.get(key, 'gal')}", "miles": f"miles_{key}", "mpg": f"mpg_{key}"}[section]
        out[name] = raw[c]
        last = name
    return out


def _read_fuel_csv(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype=str)
    ident = {
        "agency": "agency", "city": "city", "state": "state", "ntd_id": "ntd_id",
        "organization_type": "organization_type", "reporter_type": "reporter_type",
        "report_year": "report_year_in_file", "uza_name": "uza_name", "primary_uza_population": "uza_population",
        "agency_voms": "agency_voms", "modecd": "mode", "typeofservicecd": "tos", "mode_voms": "mode_voms",
    }
    out = pd.DataFrame(index=raw.index)
    last = None
    for c in raw.columns:
        if c in ident:
            out[ident[c]] = raw[c]
            continue
        if c.endswith("_questionable") or re.search(r"_1$", c):
            if last is not None:
                out[f"{last}_flag"] = raw[c]
            continue
        key = _fuel_key(c)
        if key is None:
            continue
        if c.endswith(("_mpg", "_mpkg_", "_mi_kwh")):
            name = f"mpg_{key}"
        elif c.endswith(("_gal", "_gal_gal_equivalent", "_kg_", "_kwh")):
            name = f"{key}_{FUEL_UNITS.get(key, 'gal')}"
        else:
            name = f"miles_{key}"
        out[name] = raw[c]
        last = name
    return out


def load_fuel_energy() -> tuple[pd.DataFrame, dict]:
    frames, sources = [], {}
    for year, path in fuel_energy_files().items():
        df = _read_fuel_csv(path) if path.suffix == ".csv" else _read_fuel_excel(path)
        df["report_year"] = year
        df["source_file"] = str(path.relative_to(ROOT))
        sources[str(path.relative_to(ROOT))] = sha256(path)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    df["ntd_id"] = norm_ntd_id(df["ntd_id"])
    df = df[df["ntd_id"].notna() & df["mode"].notna()].copy()
    for c in ["mode", "tos", "agency", "city", "state", "organization_type", "reporter_type"]:
        if c in df:
            df[c] = df[c].astype("string").str.strip()
    num_cols = [c for c in df.columns if c in QTY_COLS or c.startswith(("miles_", "mpg_"))] + [
        "uza_population", "agency_voms", "mode_voms", "legacy_ntd_id", "report_year_in_file"]
    for c in num_cols:
        if c in df:
            df[c] = to_num(df[c])
    flag_cols = [c for c in df.columns if c.endswith("_flag")]
    for c in flag_cols:
        s = df[c].astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA})
        # in the 2022 CSV some "_1" columns repeat the value instead of a Q/W flag: keep only real flags
        df[c] = s.where(pd.to_numeric(s, errors="coerce").isna(), pd.NA)
    df["any_flag"] = df[flag_cols].notna().any(axis=1)
    # make sure every canonical column exists in every year (NaN where the year lacks it)
    for c in QTY_COLS + [f"miles_{f}" for f in MILES_FUELS]:
        if c not in df:
            df[c] = np.nan
    return df.reset_index(drop=True), sources


# ---------------------------------------------------------------- Revenue Vehicle Inventory


def rvi_files() -> dict[int, Path]:
    out = {}
    for p in sorted(RVI_DIR.glob("*.xlsx")):
        m = re.match(r"(\d{4})", p.name)
        if m:
            out[int(m.group(1))] = p
    return out


RVI_RENAME = {
    "5 Digit NTD ID": "ntd_id", "NTD ID": "ntd_id", "Agency Name": "agency", "Reporter Type": "reporter_type",
    "Reporting Module": "reporting_module", "Form Module Type": "reporting_module",
    "Revenue Vehicle Inventory ID": "rvi_id", "Total Fleet Vehicles": "total_fleet",
    "Active Fleet Vehicles": "active_fleet", "Dedicated Fleet": "dedicated_fleet", "Vehicle Type": "vehicle_type",
    "Manufacture Year": "manufacture_year", "Rebuild Year": "rebuild_year", "Fuel Type": "fuel_type",
    "Vehicle Length": "vehicle_length_ft", "Seating Capacity": "seating_capacity",
    "Standing Capacity": "standing_capacity", "Total Miles on Active Vehicles During Period": "miles_period",
    "Average Lifetime Miles per Active Vehicles": "avg_lifetime_miles",
    "ModeTOS Vehicles Operated in Maximum Service": "modetos_voms", "Emergency Contingency Vehicles": "contingency",
    "Mode": "mode", "TOS": "tos", "Modes": "modes",
}


def load_vehicle_inventory() -> tuple[pd.DataFrame, dict]:
    """One row per fleet record and mode/TOS it serves.

    From 2018 a fleet record can serve several modes ("DR/DO, MB/DO"). Such records are
    repeated once per mode with ``n_modes`` set, so callers can either dedupe on
    (report_year, ntd_id, rvi_id) or split vehicles across modes.
    """
    frames, sources = [], {}
    for year, path in rvi_files().items():
        raw = pd.read_excel(path, engine="openpyxl", dtype=object)
        raw.columns = [str(c).strip() for c in raw.columns]
        df = raw[[c for c in raw.columns if c in RVI_RENAME]].rename(columns=RVI_RENAME)
        df = df.loc[:, ~df.columns.duplicated()]
        if "modes" not in df:
            df["modes"] = df["mode"].astype(str).str.strip() + "/" + df["tos"].astype(str).str.strip()
        df["report_year"] = year
        sources[str(path.relative_to(ROOT))] = sha256(path)
        frames.append(df.drop(columns=[c for c in ("mode", "tos") if c in df]))
    df = pd.concat(frames, ignore_index=True)
    df["ntd_id"] = norm_ntd_id(df["ntd_id"])
    df = df[df["ntd_id"].notna()].copy()
    df["modes"] = df["modes"].astype("string").str.replace(" ", "", regex=False)
    df["mode_tos"] = df["modes"].str.split(",")
    df["n_modes"] = df["mode_tos"].str.len()
    df = df.explode("mode_tos")
    parts = df["mode_tos"].str.split("/", n=1, expand=True)
    df["mode"], df["tos"] = parts[0], parts[1]
    for c in ["total_fleet", "active_fleet", "manufacture_year", "rebuild_year", "vehicle_length_ft",
              "seating_capacity", "standing_capacity", "miles_period", "avg_lifetime_miles", "modetos_voms",
              "contingency", "rvi_id"]:
        if c in df:
            df[c] = to_num(df[c])
    for c in ["agency", "reporter_type", "reporting_module", "vehicle_type", "fuel_type", "dedicated_fleet", "mode", "tos"]:
        df[c] = df[c].astype("string").str.strip()
    return df.drop(columns=["modes"]).reset_index(drop=True), sources


# ---------------------------------------------------------------- TS2.1 service time series

SERVICE_SHEETS = {"VOMS": "voms", "VRM": "vrm", "VRH": "vrh", "UPT": "upt"}


def load_service() -> tuple[pd.DataFrame, dict]:
    frames, sources = [], {}
    for path in sorted(SERVICE_DIR.glob("*.xlsx")):
        sources[str(path.relative_to(ROOT))] = sha256(path)
        sheets = pd.read_excel(path, sheet_name=list(SERVICE_SHEETS), engine="openpyxl")
        for sheet, metric in SERVICE_SHEETS.items():
            raw = sheets[sheet]
            raw.columns = [str(c).strip() for c in raw.columns]
            year_cols = {c: int(re.match(r"(\d{4})", c).group(1)) for c in raw.columns if re.match(r"^\d{4}( |$)", c)}
            keys = {"NTD ID": "ntd_id", "Agency Name": "agency", "Mode": "mode", "Type of Service": "tos",
                    "Service": "tos", "Reporter Type": "reporter_type", "Mode_Status": "mode_status",
                    "Mode Status": "mode_status", "UZA Population": "uza_population", "State": "state"}
            k = raw[[c for c in raw.columns if c in keys]].rename(columns=keys)
            long = pd.concat([k, raw[list(year_cols)].rename(columns=year_cols)], axis=1).melt(
                id_vars=list(k.columns), var_name="year", value_name="value")
            long["metric"] = metric
            long["source_file"] = path.name
            frames.append(long)
    df = pd.concat(frames, ignore_index=True)
    df["ntd_id"] = norm_ntd_id(df["ntd_id"])
    df["value"] = to_num(df["value"])
    df = df[df["ntd_id"].notna()]
    for c in ["mode", "tos", "agency", "reporter_type", "mode_status", "state"]:
        df[c] = df[c].astype("string").str.strip()
    wide = (df.groupby(["ntd_id", "mode", "tos", "year", "metric"], dropna=False)["value"]
            .sum(min_count=1).unstack("metric").reset_index())
    wide.columns.name = None
    attrs = (df.sort_values("year").groupby(["ntd_id", "mode", "tos"], as_index=False)
             [["agency", "reporter_type", "mode_status", "state", "uza_population"]].last())
    wide = wide.merge(attrs, on=["ntd_id", "mode", "tos"], how="left")
    wide["year"] = wide["year"].astype(int)
    return wide, sources


# ---------------------------------------------------------------- build


def build_interim(verbose: bool = True) -> dict[str, Path]:
    out = {}
    for name, loader in [("fuel_energy", load_fuel_energy), ("vehicle_inventory", load_vehicle_inventory),
                         ("service", load_service)]:
        df, sources = loader()
        path = INTERIM / f"{name}.parquet"
        _write(df, path, sources)
        out[name] = path
        if verbose:
            print(f"{name:18s} {len(df):>9,} rows  {len(sources)} source files -> {path.relative_to(ROOT)}")
    return out


if __name__ == "__main__":
    build_interim()
