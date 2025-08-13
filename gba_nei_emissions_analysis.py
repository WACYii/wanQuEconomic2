#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from docx import Document

sns.set(style="whitegrid", font="DejaVu Sans")

# 工作目录与输出路径
WORKDIR = Path("/workspace")
OUTPUT_DIR = WORKDIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FONTS_DIR = WORKDIR / "fonts"
FONTS_DIR.mkdir(parents=True, exist_ok=True)
CHS_FONT_PATH = FONTS_DIR / "NotoSansSC-Regular.otf"

# 文件路径
FILE_CITY_EMISSIONS = WORKDIR / "1997-2019年290个中国城市碳排放清单.xlsx"
FILE_APPARENT = WORKDIR / "表观碳排放清单_1997-2021.xlsx"
FILE_NEI_GBA = WORKDIR / "粤港澳大湾区新能源产业规模 的副本.xls"

# 九市名单（标准化后的城市名）
GBA_CITIES_CN = ["广州", "深圳", "珠海", "佛山", "惠州", "东莞", "中山", "江门", "肇庆"]


# ------------------------ 工具函数 ------------------------

def strip_zwsp(text: str) -> str:
    if text is None:
        return text
    return re.sub("\u200b", "", str(text)).strip()


def ensure_chinese_font() -> Optional[str]:
    try:
        from matplotlib import font_manager
        if not CHS_FONT_PATH.exists():
            import urllib.request
            url = (
                "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/"
                "NotoSansSC-Regular.otf"
            )
            try:
                urllib.request.urlretrieve(url, CHS_FONT_PATH)
            except Exception:
                return None
        font_manager.fontManager.addfont(str(CHS_FONT_PATH))
        plt.rcParams["font.sans-serif"] = ["Noto Sans SC", "NotoSansSC", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        return str(CHS_FONT_PATH)
    except Exception:
        return None


def normalize_city_cn(name: str) -> str:
    s = strip_zwsp(name)
    # 去省名前缀
    for prov in [
        "广东", "广西", "江苏", "浙江", "山东", "河南", "河北", "湖南", "湖北", "江西", "福建", "云南", "贵州", "四川",
        "安徽", "山西", "辽宁", "吉林", "黑龙江", "内蒙古", "青海", "甘肃", "陕西", "宁夏", "新疆", "海南",
        "北京", "天津", "上海", "重庆",
    ]:
        if s.startswith(prov):
            s = s[len(prov) :]
            break
    # 去城市后缀
    for suf in ["市", "地区", "盟", "自治州", "特别行政区", "区"]:
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(r"[^0-9eE+\-.]", "", regex=True), errors="coerce")


def safe_log(series: pd.Series, eps: float = 1e-6) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return np.log(s.clip(lower=eps))


# ------------------------ 读数函数 ------------------------

def load_city_emissions() -> pd.DataFrame:
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
    agg = (
        df_city.groupby("year", as_index=False)["emission"].sum().rename(columns={"emission": "emission_gba"})
    )
    return agg.sort_values("year")


def load_apparent_emissions_guangdong() -> pd.DataFrame:
    xls = pd.ExcelFile(FILE_APPARENT, engine="openpyxl")
    years = [s for s in xls.sheet_names if re.fullmatch(r"\d{4}", str(s))]
    records = []
    for y in years:
        dfy = pd.read_excel(xls, sheet_name=y)
        dfy.columns = [strip_zwsp(c) for c in dfy.columns]
        col_gd = None
        for cand in ["Guangdong", "广东", "Guang Dong"]:
            if cand in dfy.columns:
                col_gd = cand
                break
        if not col_gd or "Items" not in dfy.columns:
            continue
        mask = dfy["Items"].isna() & dfy[col_gd].notna()
        if mask.any():
            val = float(pd.to_numeric(dfy.loc[mask, col_gd]).astype(float).iloc[0])
            records.append({"year": int(y), "emission_guangdong": val})
    return pd.DataFrame.from_records(records).sort_values("year")


def load_gba_nei() -> pd.DataFrame:
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
    # 容错匹配
    for key in list(rename_map.keys()):
        if key not in df.columns:
            for col in df.columns:
                if key.replace("（", "(").replace("）", ")") in col.replace("（", "(").replace("）", ")"):
                    rename_map[col] = rename_map.pop(key)
                    break
    df = df.rename(columns=rename_map)
    for col in ["year", "re_total_gw", "pv_gw", "wind_gw", "nev_10k_units", "storage_mw"]:
        if col in df.columns:
            df[col] = to_numeric(df[col])
    df = df[~df["year"].isna()].copy()
    df["year"] = df["year"].astype(int)
    return df.sort_values("year")


# ------------------------ 建模辅助 ------------------------

def fit_time_series_ols(
    df: pd.DataFrame, y: str, x_vars: List[str], trend: bool = True, cluster: Optional[pd.Series] = None
) -> sm.regression.linear_model.RegressionResultsWrapper:
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


# ------------------------ 图表与导出 ------------------------

def plot_and_export(df_gba_year: pd.DataFrame, df_nei: pd.DataFrame, merged_gba: pd.DataFrame, panel: pd.DataFrame, outputs):
    ensure_chinese_font()

    # 图1：九市碳排放总量
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(df_gba_year["year"], df_gba_year["emission_gba"], marker="o", color="#2E86DE")
    ax1.axvspan(2014, 2019, color="#2E86DE", alpha=0.08, label="与新能源指标重叠区间")
    ax1.set_title("粤港澳大湾区九市碳排放总量（1997-2019）")
    ax1.set_xlabel("年份")
    ax1.set_ylabel("碳排放（原单位）")
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(OUTPUT_DIR / "图1_九市碳排放总量_1997_2019.png", dpi=200)
    plt.close(fig1)

    # 图2：新能源装机规模
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(df_nei["year"], df_nei["pv_gw"], marker="o", label="光伏装机（GW）")
    ax2.plot(df_nei["year"], df_nei["wind_gw"], marker="o", label="风电装机（GW）")
    ax2.plot(df_nei["year"], df_nei["re_total_gw"], marker="o", label="可再生能源装机总量（GW）")
    ax2.set_title("大湾区新能源装机规模（2014-2023）")
    ax2.set_xlabel("年份")
    ax2.set_ylabel("装机规模（GW）")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(OUTPUT_DIR / "图2_新能源装机规模_2014_2023.png", dpi=200)
    plt.close(fig2)

    # 图3：新能源汽车与储能
    fig3, ax3a = plt.subplots(figsize=(10, 6))
    ax3a.plot(df_nei["year"], df_nei["nev_10k_units"], color="#27AE60", marker="s", label="新能源汽车保有量（万辆）")
    ax3a.set_xlabel("年份")
    ax3a.set_ylabel("新能源汽车保有量（万辆）", color="#27AE60")
    ax3a.tick_params(axis="y", labelcolor="#27AE60")
    ax3b = ax3a.twinx()
    ax3b.plot(df_nei["year"], df_nei["storage_mw"], color="#8E44AD", marker="^", label="储能装机（MW）")
    ax3b.set_ylabel("储能装机（MW）", color="#8E44AD")
    ax3b.tick_params(axis="y", labelcolor="#8E44AD")
    ax3a.set_title("大湾区新能源汽车与储能发展（2014-2023）")
    fig3.tight_layout()
    fig3.savefig(OUTPUT_DIR / "图3_NEVE与储能_2014_2023.png", dpi=200)
    plt.close(fig3)

    # 图4：散点关系
    if not merged_gba.empty:
        labels_map = {
            "pv_gw": "光伏装机（GW）",
            "wind_gw": "风电装机（GW）",
            "nev_10k_units": "新能源汽车保有量（万辆）",
            "storage_mw": "储能装机（MW）",
        }
        for var, label in labels_map.items():
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.regplot(data=merged_gba, x=var, y="emission_gba", ax=ax, marker="o", color="#2C3E50")
            ax.set_title(f"九市碳排放与{label}的关系（2014-2019）")
            ax.set_xlabel(label)
            ax.set_ylabel("九市碳排放（原单位）")
            fig.tight_layout()
            fig.savefig(OUTPUT_DIR / f"图4_散点_{var}_vs_九市碳排放_2014_2019.png", dpi=200)
            plt.close(fig)

    # 图5：双轴图
    if not merged_gba.empty:
        fig4, ax4a = plt.subplots(figsize=(10, 6))
        ax4a.plot(merged_gba["year"], merged_gba["emission_gba"], color="#e67e22", marker="o", label="九市碳排放")
        ax4a.set_xlabel("年份")
        ax4a.set_ylabel("九市碳排放（原单位）", color="#e67e22")
        ax4a.tick_params(axis="y", labelcolor="#e67e22")
        ax4b = ax4a.twinx()
        ax4b.plot(merged_gba["year"], merged_gba["re_total_gw"], color="#2980b9", marker="s", label="可再生能源装机总量（GW）")
        ax4b.set_ylabel("可再生能源装机总量（GW）", color="#2980b9")
        ax4b.tick_params(axis="y", labelcolor="#2980b9")
        ax4a.set_title("九市碳排放与可再生能源装机（2014-2019）")
        fig4.tight_layout()
        fig4.savefig(OUTPUT_DIR / "图5_双轴_九市碳排放_vs_可再生装机_2014_2019.png", dpi=200)
        plt.close(fig4)

    # 图6：城市趋势
    if not panel.empty:
        fig5, ax5 = plt.subplots(figsize=(10, 6))
        for c, sub in panel.groupby("city_std"):
            ax5.plot(sub["year"], sub["emission"], marker="o", label=c)
        ax5.set_title("九市城市碳排放趋势（2014-2019）")
        ax5.set_xlabel("年份")
        ax5.set_ylabel("碳排放（原单位）")
        ax5.legend(ncols=3, fontsize=8)
        fig5.tight_layout()
        fig5.savefig(OUTPUT_DIR / "图6_九市城市碳排放趋势_2014_2019.png", dpi=200)
        plt.close(fig5)

    # 导出回归结果（CSV与Word）
    def tidy_result(res: sm.regression.linear_model.RegressionResultsWrapper, name: str) -> pd.DataFrame:
        params = res.params
        se = res.bse
        conf = res.conf_int()
        pvals = res.pvalues
        out = pd.DataFrame(
            {
                "变量": params.index,
                "系数": params.values,
                "标准误": se.values,
                "下界95%": conf[0].values,
                "上界95%": conf[1].values,
                "p值": pvals.values,
            }
        )
        out.insert(0, "模型", name)
        return out

    all_tidy = []
    for name, res in outputs:
        all_tidy.append(tidy_result(res, name))
    if all_tidy:
        tidy_df = pd.concat(all_tidy, ignore_index=True)
        tidy_df.to_csv(OUTPUT_DIR / "回归结果_可粘贴.csv", index=False)
        doc = Document()
        doc.add_heading("回归结果汇总", level=1)
        for name, res in outputs:
            doc.add_heading(name, level=2)
            tbl = doc.add_table(rows=1, cols=6)
            hdr = tbl.rows[0].cells
            hdr[0].text = "变量"; hdr[1].text = "系数"; hdr[2].text = "标准误"; hdr[3].text = "下界95%"; hdr[4].text = "上界95%"; hdr[5].text = "p值"
            params = res.params
            se = res.bse
            conf = res.conf_int()
            p = res.pvalues
            for var in params.index:
                row = tbl.add_row().cells
                row[0].text = str(var)
                row[1].text = f"{params[var]:.4f}"
                row[2].text = f"{se[var]:.4f}"
                row[3].text = f"{conf.loc[var, 0]:.4f}"
                row[4].text = f"{conf.loc[var, 1]:.4f}"
                row[5].text = f"{p[var]:.4f}"
        doc.save(OUTPUT_DIR / "回归结果汇总.docx")


# ------------------------ 主流程 ------------------------

def main():
    # 读取
    df_city = load_city_emissions()
    df_gba_year = aggregate_gba_emissions_city_to_year(df_city)
    df_gd_prov = load_apparent_emissions_guangdong()
    df_nei = load_gba_nei()

    # 中间文件
    df_city.to_csv(OUTPUT_DIR / "city_emissions_gba_1997_2019.csv", index=False)
    df_gba_year.to_csv(OUTPUT_DIR / "gba_total_emissions_1997_2019.csv", index=False)
    df_gd_prov.to_csv(OUTPUT_DIR / "guangdong_apparent_emissions_1997_2021.csv", index=False)
    df_nei.to_csv(OUTPUT_DIR / "gba_nei_2014_2023.csv", index=False)

    # 匹配窗口
    merged_gba = df_gba_year.merge(df_nei, on="year", how="inner").query("year >= 2014 and year <= 2019")
    merged_gd = df_gd_prov.merge(df_nei, on="year", how="inner").query("year >= 2014 and year <= 2021")
    panel = df_city.query("year >= 2014 and year <= 2019").copy()

    # 回归
    outputs: List[Tuple[str, sm.regression.linear_model.RegressionResultsWrapper]] = []

    # 九市时间序列
    if len(merged_gba) >= 5:
        dfg = merged_gba.copy()
        dfg["ln_emission_gba"] = safe_log(dfg["emission_gba"])
        for v in ["pv_gw", "wind_gw", "nev_10k_units", "storage_mw"]:
            dfg[f"ln_{v}"] = safe_log(dfg[v])
        xvars = ["ln_pv_gw", "ln_wind_gw", "ln_nev_10k_units", "ln_storage_mw"]
        res_ts = fit_time_series_ols(dfg, y="ln_emission_gba", x_vars=xvars, trend=True)
        outputs.append(("九市-时间序列（2014-2019）", res_ts))

    # 广东省时间序列
    if len(merged_gd) >= 6:
        dfgd = merged_gd.copy()
        dfgd["ln_emission_gd"] = safe_log(dfgd["emission_guangdong"])
        for v in ["pv_gw", "wind_gw", "nev_10k_units", "storage_mw"]:
            dfgd[f"ln_{v}"] = safe_log(dfgd[v])
        xvars = ["ln_pv_gw", "ln_wind_gw", "ln_nev_10k_units", "ln_storage_mw"]
        res_gd = fit_time_series_ols(dfgd, y="ln_emission_gd", x_vars=xvars, trend=True)
        outputs.append(("广东-时间序列（2014-2021）", res_gd))

    # 城市固定效应
    panel_merged = panel.merge(df_nei, on="year", how="left")
    if len(panel_merged) >= 30:
        dfp = panel_merged.copy()
        dfp["ln_emission_city"] = safe_log(dfp["emission"])
        for v in ["pv_gw", "wind_gw", "nev_10k_units", "storage_mw"]:
            dfp[f"ln_{v}"] = safe_log(dfp[v])
        dfp, city_dummies = add_city_dummies(dfp, "city_std")
        xvars = ["ln_pv_gw", "ln_wind_gw", "ln_nev_10k_units", "ln_storage_mw"] + city_dummies
        res_fe = fit_time_series_ols(dfp, y="ln_emission_city", x_vars=xvars, trend=True, cluster=dfp["year"])  # 按年份聚类稳健SE
        outputs.append(("九市-城市固定效应（2014-2019）", res_fe))

    # 文本回归结果
    with open(OUTPUT_DIR / "regression_results.txt", "w", encoding="utf-8") as f:
        for name, res in outputs:
            f.write("=" * 100 + "\n"); f.write(name + "\n"); f.write("-" * 100 + "\n")
            f.write(res.summary().as_text()); f.write("\n\n")

    # 图表与Word表格
    plot_and_export(df_gba_year, df_nei, merged_gba, panel, outputs)

    print("已保存输出至:", OUTPUT_DIR)


if __name__ == "__main__":
    ensure_chinese_font()
    main()