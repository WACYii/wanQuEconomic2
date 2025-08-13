#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import re
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", font="DejaVu Sans")

WORKDIR = Path("/workspace")
OUTPUT_DIR = WORKDIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# File paths
FILE_CITY_EMISSIONS = WORKDIR / "1997-2019年290个中国城市碳排放清单.xlsx"
FILE_APPARENT = WORKDIR / "表观碳排放清单_1997-2021.xlsx"
FILE_NEI_GBA = WORKDIR / "粤港澳大湾区新能源产业规模 的副本.xls"

GBA_CITIES_CN = [
    "广州", "深圳", "珠海", "佛山", "惠州",
    "东莞", "中山", "江门", "肇庆",
]

# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------

def strip_zwsp(text: str) -> str:
    if text is None:
        return text
    # remove zero-width spaces and surrounding whitespace
    return re.sub("\u200b", "", str(text)).strip()


def normalize_city_cn(name: str) -> str:
    """Normalize Chinese city names by stripping province prefixes and common suffixes.

    Examples:
    - '广东广州' -> '广州'
    - '广州市' -> '广州'
    - '广东深圳市' -> '深圳'
    """
    s = strip_zwsp(name)
    # remove province prefix if present (common pattern '广东')
    for prov in ["广东", "广西", "江苏", "浙江", "山东", "河南", "河北", "湖南", "湖北", "江西", "福建", "云南", "贵州", "四川", "安徽", "山西", "辽宁", "吉林", "黑龙江", "内蒙古", "青海", "甘肃", "陕西", "宁夏", "新疆", "海南", "北京", "天津", "上海", "重庆"]:
        if s.startswith(prov):
            s = s[len(prov):]
            break
    # remove common suffixes
    for suf in ["市", "地区", "盟", "自治州", "特别行政区", "区"]:
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(
        r"[^0-9eE+\-.]", "", regex=True
    ), errors="coerce")


def safe_log(series: pd.Series, eps: float = 1e-6) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return np.log(s.clip(lower=eps))


# --------------------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------------------

def load_city_emissions() -> pd.DataFrame:
    """Load city-year emissions (Mt CO2 or similar), filter to 9 GBA cities.

    Returns
    -------
    DataFrame with columns: city, year, emission
    """
    xls = pd.ExcelFile(FILE_CITY_EMISSIONS, engine="openpyxl")
    df = pd.read_excel(xls, sheet_name="emission vector")
    df.columns = [c.strip() for c in df.columns]
    df = df[["city", "year", "emission"]]
    df["city"] = df["city"].astype(str).str.strip()
    df["city_std"] = df["city"].apply(normalize_city_cn)
    df = df[df["city_std"].isin(GBA_CITIES_CN)].copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
    df["emission"] = pd.to_numeric(df["emission"], errors="coerce")
    return df


def aggregate_gba_emissions_city_to_year(df_city: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 9-city emissions to GBA yearly total.

    Returns columns: year, emission_gba
    """
    agg = (
        df_city.groupby("year", as_index=False)["emission"].sum()
        .rename(columns={"emission": "emission_gba"})
        .sort_values("year")
    )

    return agg


def load_apparent_emissions_guangdong() -> pd.DataFrame:
    """Extract Guangdong provincial 'apparent carbon emissions' by year.

    The Excel sheets contain several items per year; the total emissions row
    appears as a blank/NaN 'Items' cell (due to merged cells). We detect the
    first row with NaN in 'Items' but numeric in 'Guangdong'.
    """
    xls = pd.ExcelFile(FILE_APPARENT, engine="openpyxl")
    years = [s for s in xls.sheet_names if re.fullmatch(r"\d{4}", str(s))]
    records = []
    for y in years:
        dfy = pd.read_excel(xls, sheet_name=y)
        # Normalize columns
        dfy.columns = [strip_zwsp(c) for c in dfy.columns]
        col_gd = None
        for cand in ["Guangdong", "广东", "Guang Dong"]:
            if cand in dfy.columns:
                col_gd = cand
                break
        if not col_gd:
            continue
        # Find the first NaN Items row with numeric Guangdong value
        items = dfy.get("Items")
        if items is None:
            continue
        mask = items.isna() & dfy[col_gd].notna()
        if mask.any():
            val = pd.to_numeric(dfy.loc[mask, col_gd]).astype(float).iloc[0]
            records.append({"year": int(y), "emission_guangdong": float(val)})
        else:
            # Fallback: try to locate row after 'Cement'
            idxs = dfy.index[dfy["Items"].astype(str).str.contains(
                "Cement", case=False, na=False
            )].tolist()
            if idxs:
                idx = idxs[0] + 1
                if idx < len(dfy) and pd.notna(dfy.loc[idx, col_gd]):
                    val = float(dfy.loc[idx, col_gd])
                    records.append({"year": int(y), "emission_guangdong": val})
    out = pd.DataFrame.from_records(records).sort_values("year")
    return out


def load_gba_nei() -> pd.DataFrame:
    """Load GBA new energy indicators from the .xls file.

    Expected columns (with zero-width spaces removed):
    - 年份
    - 可再生能源装机总量（GW）
    - 光伏装机（GW）
    - 风电装机（GW）
    - 新能源汽车保有量（万辆）
    - 储能装机（MW）
    - 关键事件
    """
    df = pd.read_excel(FILE_NEI_GBA, sheet_name=0)
    df.columns = [strip_zwsp(c) for c in df.columns]
    rename_map = {
        "年份": "year",
        "可再生能源装机总量（GW）": "re_total_gw",
        "光伏装机（GW）": "pv_gw",
        "风电装机（GW）": "wind_gw",
        "新能源汽车保有量（万辆）": "nev_10k_units",
        "储能装机（MW）": "storage_mw",
        "关键事件": "events",
    }
    # Handle cases where full-width or alternate spaces exist
    for key in list(rename_map.keys()):
        if key not in df.columns:
            # Try relaxed matching
            for col in df.columns:
                if key.replace("（", "(").replace("）", ")") in col.replace("（", "(").replace("）", ")"):
                    rename_map[col] = rename_map.pop(key)
                    break
    df = df.rename(columns=rename_map)

    # Coerce numerics
    for col in ["year", "re_total_gw", "pv_gw", "wind_gw", "nev_10k_units", "storage_mw"]:
        if col in df.columns:
            df[col] = to_numeric(df[col])
    
    # Drop rows without year
    df = df[~df["year"].isna()].copy()
    df["year"] = df["year"].astype(int)

    # Convert NEV to units (10k units -> million units or leave as 10k)
    # Here we keep as 10k units to avoid tiny numbers; logs remain well-behaved.

    # Sort and keep plausible range
    df = df.sort_values("year")
    return df


# --------------------------------------------------------------------------------------
# Modeling helpers
# --------------------------------------------------------------------------------------

def fit_time_series_ols(df: pd.DataFrame, y: str, x_vars: List[str], trend: bool = True,
                        cluster: Optional[pd.Series] = None) -> sm.regression.linear_model.RegressionResultsWrapper:
    X = df[x_vars].copy()
    if trend:
        X["trend"] = np.arange(len(df)) + 1
    X = sm.add_constant(X)
    model = sm.OLS(df[y].astype(float), X.astype(float), missing="drop")
    if cluster is not None:
        res = model.fit(cov_type="cluster", cov_kwds={"groups": cluster})
    else:
        res = model.fit(cov_type="HAC", cov_kwds={"maxlags": 1})
    return res


def add_city_dummies(df: pd.DataFrame, city_col: str) -> Tuple[pd.DataFrame, List[str]]:
    dummies = pd.get_dummies(df[city_col], prefix="city", drop_first=True)
    return pd.concat([df, dummies], axis=1), list(dummies.columns)


# --------------------------------------------------------------------------------------
# Main analysis pipeline
# --------------------------------------------------------------------------------------

def main() -> None:
    # 1) Load datasets
    df_city = load_city_emissions()
    df_gba_year = aggregate_gba_emissions_city_to_year(df_city)
    df_gd_prov = load_apparent_emissions_guangdong()
    df_nei = load_gba_nei()

    # Save intermediates
    df_city.to_csv(OUTPUT_DIR / "city_emissions_gba_1997_2019.csv", index=False)
    df_gba_year.to_csv(OUTPUT_DIR / "gba_total_emissions_1997_2019.csv", index=False)
    df_gd_prov.to_csv(OUTPUT_DIR / "guangdong_apparent_emissions_1997_2021.csv", index=False)
    df_nei.to_csv(OUTPUT_DIR / "gba_nei_2014_2023.csv", index=False)

    # 2) Merge for overlapping windows
    # 2.a) GBA total emissions with NEI (2014-2019)
    merged_gba = (
        df_gba_year.merge(df_nei, on="year", how="inner")
        .query("year >= 2014 and year <= 2019")
        .reset_index(drop=True)
    )

    # 2.b) Guangdong provincial emissions with NEI (2014-2021)
    merged_gd = (
        df_gd_prov.merge(df_nei, on="year", how="inner")
        .query("year >= 2014 and year <= 2021")
        .reset_index(drop=True)
    )

    # 2.c) City-level panel (2014-2019)
    panel = df_city.query("year >= 2014 and year <= 2019").copy()

    # 3) Visualization
    # 3.1) Long-run GBA emissions trend (1997-2019)
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(df_gba_year["year"], df_gba_year["emission_gba"], marker="o", color="#2E86DE")
    ax1.axvspan(2014, 2019, color="#2E86DE", alpha=0.08, label="Overlap with NEI")
    ax1.set_title("GBA total CO2 emissions (1997-2019)")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Emissions (unit as in source)")
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(OUTPUT_DIR / "fig_gba_emissions_1997_2019.png", dpi=200)
    plt.close(fig1)

    # 3.2) NEI indicators (2014-2023)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(df_nei["year"], df_nei["pv_gw"], marker="o", label="PV (GW)")
    ax2.plot(df_nei["year"], df_nei["wind_gw"], marker="o", label="Wind (GW)")
    ax2.plot(df_nei["year"], df_nei["re_total_gw"], marker="o", label="RE total (GW)")
    ax2.set_title("GBA renewable capacity (2014-2023)")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("GW")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(OUTPUT_DIR / "fig_gba_nei_capacities_2014_2023.png", dpi=200)
    plt.close(fig2)

    # NEV + Storage on secondary axis
    fig3, ax3a = plt.subplots(figsize=(10, 6))
    ax3a.plot(df_nei["year"], df_nei["nev_10k_units"], color="#27AE60", marker="s", label="NEV (10k units)")
    ax3a.set_xlabel("Year")
    ax3a.set_ylabel("NEV (10k units)", color="#27AE60")
    ax3a.tick_params(axis='y', labelcolor="#27AE60")

    ax3b = ax3a.twinx()
    ax3b.plot(df_nei["year"], df_nei["storage_mw"], color="#8E44AD", marker="^", label="Storage (MW)")
    ax3b.set_ylabel("Storage (MW)", color="#8E44AD")
    ax3b.tick_params(axis='y', labelcolor="#8E44AD")

    ax3a.set_title("GBA NEV and Storage (2014-2023)")
    fig3.tight_layout()
    fig3.savefig(OUTPUT_DIR / "fig_gba_nev_storage_2014_2023.png", dpi=200)
    plt.close(fig3)

    # 3.3) Scatter: NEI vs GBA emissions (2014-2019)
    if not merged_gba.empty:
        for var, label in [("pv_gw", "PV (GW)"), ("wind_gw", "Wind (GW)"), ("nev_10k_units", "NEV (10k units)"), ("storage_mw", "Storage (MW)")]:
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.regplot(data=merged_gba, x=var, y="emission_gba", ax=ax, marker="o", color="#2C3E50")
            ax.set_title(f"GBA emissions vs {label} (2014-2019)")
            fig.tight_layout()
            fig.savefig(OUTPUT_DIR / f"fig_scatter_{var}_vs_emission_gba_2014_2019.png", dpi=200)
            plt.close(fig)

    # 4) Modeling
    outputs = []

    # 4.a) GBA-level time series (2014-2019)
    if len(merged_gba) >= 5:  # minimal points
        dfg = merged_gba.copy()
        dfg["ln_emission_gba"] = safe_log(dfg["emission_gba"])
        dfg["ln_pv_gw"] = safe_log(dfg["pv_gw"])
        dfg["ln_wind_gw"] = safe_log(dfg["wind_gw"])
        dfg["ln_nev_10k_units"] = safe_log(dfg["nev_10k_units"])
        dfg["ln_storage_mw"] = safe_log(dfg["storage_mw"])

        xvars = ["ln_pv_gw", "ln_wind_gw", "ln_nev_10k_units", "ln_storage_mw"]
        res_ts = fit_time_series_ols(dfg, y="ln_emission_gba", x_vars=xvars, trend=True)
        outputs.append(("GBA_time_series_2014_2019", res_ts))

    # 4.b) Guangdong provincial time series (2014-2021)
    if len(merged_gd) >= 6:
        dfgd = merged_gd.copy()
        dfgd["ln_emission_guangdong"] = safe_log(dfgd["emission_guangdong"])
        dfgd["ln_pv_gw"] = safe_log(dfgd["pv_gw"])
        dfgd["ln_wind_gw"] = safe_log(dfgd["wind_gw"])
        dfgd["ln_nev_10k_units"] = safe_log(dfgd["nev_10k_units"])
        dfgd["ln_storage_mw"] = safe_log(dfgd["storage_mw"])

        xvars = ["ln_pv_gw", "ln_wind_gw", "ln_nev_10k_units", "ln_storage_mw"]
        res_gd = fit_time_series_ols(dfgd, y="ln_emission_guangdong", x_vars=xvars, trend=True)
        outputs.append(("Guangdong_time_series_2014_2021", res_gd))

    # 4.c) City FE panel (2014-2019) - city dummies, cluster by year
    panel_merged = panel.merge(df_nei, on="year", how="left")
    if len(panel_merged) >= 30:
        dfp = panel_merged.copy()
        dfp["ln_emission_city"] = safe_log(dfp["emission"])
        dfp["ln_pv_gw"] = safe_log(dfp["pv_gw"])
        dfp["ln_wind_gw"] = safe_log(dfp["wind_gw"])
        dfp["ln_nev_10k_units"] = safe_log(dfp["nev_10k_units"])
        dfp["ln_storage_mw"] = safe_log(dfp["storage_mw"])
        dfp, city_dummies = add_city_dummies(dfp, "city")
        xvars = ["ln_pv_gw", "ln_wind_gw", "ln_nev_10k_units", "ln_storage_mw"] + city_dummies
        res_fe = fit_time_series_ols(dfp, y="ln_emission_city", x_vars=xvars, trend=True, cluster=dfp["year"])  # cluster by year
        outputs.append(("City_FE_panel_2014_2019", res_fe))

    # 5) Save regression summaries
    with open(OUTPUT_DIR / "regression_results.txt", "w", encoding="utf-8") as f:
        for name, res in outputs:
            f.write("=" * 100 + "\n")
            f.write(name + "\n")
            f.write("-" * 100 + "\n")
            f.write(res.summary().as_text())
            f.write("\n\n")

    # 6) Dual-axis plot for GBA merge
    if not merged_gba.empty:
        fig4, ax4a = plt.subplots(figsize=(10, 6))
        ax4a.plot(merged_gba["year"], merged_gba["emission_gba"], color="#e67e22", marker="o", label="Emissions (GBA)")
        ax4a.set_xlabel("Year")
        ax4a.set_ylabel("Emissions (GBA)", color="#e67e22")
        ax4a.tick_params(axis='y', labelcolor="#e67e22")
        ax4b = ax4a.twinx()
        ax4b.plot(merged_gba["year"], merged_gba["re_total_gw"], color="#2980b9", marker="s", label="RE total (GW)")
        ax4b.set_ylabel("RE total (GW)", color="#2980b9")
        ax4b.tick_params(axis='y', labelcolor="#2980b9")
        ax4a.set_title("GBA emissions vs RE total (2014-2019)")
        fig4.tight_layout()
        fig4.savefig(OUTPUT_DIR / "fig_dualaxis_gba_emissions_vs_re_2014_2019.png", dpi=200)
        plt.close(fig4)

    # 7) City emissions trend (2014-2019)
    if not panel.empty:
        fig5, ax5 = plt.subplots(figsize=(10, 6))
        for c, sub in panel.groupby("city_std"):
            ax5.plot(sub["year"], sub["emission"], marker="o", label=c)
        ax5.set_title("City-level emissions in GBA (2014-2019)")
        ax5.set_xlabel("Year")
        ax5.set_ylabel("Emissions")
        ax5.legend(ncols=3, fontsize=8)
        fig5.tight_layout()
        fig5.savefig(OUTPUT_DIR / "fig_city_emissions_trend_2014_2019.png", dpi=200)
        plt.close(fig5)

    print("Saved outputs to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()