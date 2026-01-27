"""
Medellín Urban Green Equity Analysis
===================================

This script implements a reproducible geospatial pipeline to assess socio-spatial 
inequities in urban greenery (tree canopy coverage) across socioeconomic strata 
(estratos) in Medellín, Colombia.

It reproduces key analyses from:
Martinez, J., Argota Sánchez-Vaquerizo, J., & Mahajan, S. (2026). 
"Not Your Mean Green: Beyond Averages in Mapping Socio-Spatial Inequities in Urban Greenery for Smart Cities"

Requirements:
- Input GeoJSON: "medellin_with_population_v2.geojson"
  Must contain: geometry, estrato or ESTRATO (int), pop_2025 (float), canopy_coverage (float)

Outputs:
- Console: Statistical summaries (ANOVA, Spearman, Moran’s I, etc.)
- Figure: medellin_four_panel_figure.png (4-panel equity visualization)

Author: Sachit Mahajan (ETH Zurich)
License: MIT
"""

import warnings
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import kruskal, levene, chi2_contingency
from libpysal.weights import Queen, KNN
from esda.moran import Moran

# ---------------------------
# Configuration
# ---------------------------
warnings.filterwarnings("ignore", category=UserWarning, module="libpysal")
EPS = 1e-6                      # Avoid division by zero
AREA_CRS = "EPSG:6933"          # Equal-area projection (World Cylindrical Equal Area)
CITY_CRS = "EPSG:3115"          # Medellín local projected CRS (MAGNA-SIRGAS zone)
INPUT_FILE = "medellin_with_population_v2.geojson"
OUTPUT_FIG = "medellin_four_panel_figure.png"

# ---------------------------
# 1. Load and preprocess data
# ---------------------------
print("📥 Loading and preprocessing data...")
gdf_raw = gpd.read_file(INPUT_FILE)

# Standardize column name
if "estrato" in gdf_raw.columns and "ESTRATO" not in gdf_raw.columns:
    gdf_raw = gdf_raw.rename(columns={"estrato": "ESTRATO"})

gdf = gdf_raw.to_crs(AREA_CRS)
gdf["ESTRATO"] = gdf["ESTRATO"].astype(int)
gdf["area_km2"] = gdf.geometry.area / 1e6
gdf["pop_density"] = gdf["pop_2025"] / gdf["area_km2"].replace(0, np.nan)

# Remove sliver artifacts
sliver_mask = (gdf["area_km2"] < 0.01) & (gdf["pop_density"] > 200_000)
if sliver_mask.any():
    print(f"⚠️  Removing {sliver_mask.sum()} sliver polygons "
          f"({gdf.loc[sliver_mask, 'pop_2025'].sum():,.0f} people)")
    gdf = gdf.loc[~sliver_mask].copy()
else:
    print("✅ No slivers detected.")

# Create city-projection copy for spatial weights
gdf_city = gdf.to_crs(CITY_CRS)
gdf_city["geometry"] = gdf_city.geometry.buffer(0)
assert gdf_city.is_valid.all()

# Derived variables
gdf["canopy_per_capita"] = gdf["canopy_coverage"] / (gdf["pop_2025"] + EPS)
gdf["canopy_category"] = pd.cut(
    gdf["canopy_coverage"], bins=[-1, 10, 30, 100], labels=["Low", "Medium", "High"]
)
gdf["hi_pop_lo_can"] = ((gdf["canopy_coverage"] < 10) & (gdf["pop_2025"] > 500)).astype(int)

# ---------------------------
# 2. Descriptive & inferential statistics
# ---------------------------
print(f"\n📊 Basic info — {len(gdf)} polygons; estratos {sorted(gdf['ESTRATO'].unique())}")
print(f"Total population: {gdf['pop_2025'].sum():,.0f}")

# Population-weighted mean canopy by estrato
w_means = gdf.groupby("ESTRATO").apply(
    lambda t: np.average(t["canopy_coverage"], weights=t["pop_2025"] + EPS)
)
print("\n1️⃣  Pop-weighted mean canopy (%):\n", w_means.round(2))

# Spearman correlation
rho, p = stats.spearmanr(gdf["ESTRATO"], gdf["canopy_coverage"], nan_policy="omit")
print(f"\n2️⃣  Spearman ρ = {rho:.3f}, p = {p:.3f}")

# ANOVA + Tukey HSD
anova_df = gdf.dropna(subset=["canopy_coverage"])
if not anova_df.empty:
    aov = ols("canopy_coverage ~ C(ESTRATO)", data=anova_df).fit()
    print("\n3️⃣  ANOVA (canopy ~ estrato):\n", sm.stats.anova_lm(aov, typ=2))
    print("\n4️⃣  Tukey HSD:\n", pairwise_tukeyhsd(anova_df["canopy_coverage"], anova_df["ESTRATO"]))

# Kruskal-Wallis & Levene
groups = [grp.dropna() for _, grp in gdf.groupby("ESTRATO")["canopy_coverage"]]
groups = [g for g in groups if len(g) > 0]
if len(groups) > 1:
    kw = kruskal(*groups)
    lv = levene(*groups)
    print(f"\n5️⃣  Kruskal H = {kw.statistic:.1f}, p = {kw.pvalue:.3f}")
    print(f"6️⃣  Levene W = {lv.statistic:.1f}, p = {lv.pvalue:.3f}")

# Chi-square test
chi_df = gdf.dropna(subset=["canopy_coverage"]).copy()
ct = pd.crosstab(chi_df["ESTRATO"], chi_df["canopy_category"])
if ct.shape[0] > 1 and ct.shape[1] > 1:
    chi2, p, dof, _ = chi2_contingency(ct)
    print(f"\n7️⃣  χ² = {chi2:.1f}, dof = {dof}, p = {p:.3f}")
    print(ct)

# Moran’s I (spatial autocorrelation)
print("\n🔗 Moran’s I (Queen → KNN fallback)")
data_m = gdf_city.dropna(subset=["canopy_coverage"]).copy()
if len(data_m) >= 3:
    w = Queen.from_dataframe(data_m, silence_warnings=True)
    if w.n_components > len(data_m) * 0.05:
        k = min(8, len(data_m) - 1)
        w = KNN.from_dataframe(data_m, k=k, silence_warnings=True)
        print(f"→ Using KNN (k={k}) due to disconnected components")
    w.transform = "r"
    moran = Moran(data_m["canopy_coverage"].values, w)
    print(f"I = {moran.I:.3f}, z = {moran.z_sim:.1f}, p = {moran.p_sim:.3f}")
else:
    print("Not enough valid polygons for spatial analysis.")

# High-risk zones
hotspot_count = gdf["hi_pop_lo_can"].sum()
hotspot_pop = gdf.loc[gdf["hi_pop_lo_can"] == 1, "pop_2025"].sum()
print(f"\n9️⃣  High-risk zones: {hotspot_count} polygons ({hotspot_pop:,.0f} people)")

# ---------------------------
# 3. Generate 4-panel figure
# ---------------------------
print("\n🖼️  Generating 4-panel equity figure...")

sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams.update({
    "font.size": 24, "axes.titlesize": 26, "axes.labelsize": 24,
    "xtick.labelsize": 22, "ytick.labelsize": 22, "legend.fontsize": 22,
    "axes.titlepad": 18
})

fig, axes = plt.subplots(2, 2, figsize=(20, 14))
estratos = sorted(gdf["ESTRATO"].unique())
rng = np.random.default_rng(0)

# (a) Boxplot
sns.boxplot(x="ESTRATO", y="canopy_coverage", data=gdf, ax=axes[0, 0])
axes[0, 0].set_xlabel("Estrato")
axes[0, 0].set_ylabel("Canopy Coverage (%)")

# (b) Pop-weighted means with bootstrap CI
means, lo_err, hi_err, sizes = [], [], [], []
for e, grp in gdf.groupby("ESTRATO"):
    x = grp["canopy_coverage"].to_numpy()
    w = (grp["pop_2025"] + EPS).to_numpy()
    mu = np.average(x, weights=w)
    means.append(mu)
    sizes.append(len(grp))
    p = w / w.sum()
    B = 2000
    idx = rng.choice(len(x), size=(B, len(x)), replace=True, p=p)
    boot = x[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    lo_err.append(mu - lo)
    hi_err.append(hi - mu)

axes[0, 1].bar(range(1, 7), means, yerr=[lo_err, hi_err], capsize=6, width=0.7)
axes[0, 1].set_xlabel("Estrato")
axes[0, 1].set_ylabel("Pop-weighted Mean Canopy (%)")
for i, n in enumerate(sizes):
    axes[0, 1].text(i + 1, 0.1, f"n={n}", ha="center", fontsize=16)

# (c) Stacked bar (canopy categories)
cat_tbl = (pd.crosstab(gdf["ESTRATO"], gdf["canopy_category"], normalize="index") * 100)[["Low", "Medium", "High"]]
cat_tbl.plot(kind="barh", stacked=True, colormap="viridis", ax=axes[1, 0], legend=False)
axes[1, 0].set_xlabel("Share of Polygons (%)")
axes[1, 0].set_ylabel("Estrato")
handles = [plt.Rectangle((0, 0), 1, 1, color=plt.get_cmap("viridis")(v)) for v in [0, 0.5, 1]]
axes[1, 0].legend(handles, ["Low", "Medium", "High"], title="Canopy Category", loc="lower left", title_fontsize=20)

# (d) Tukey HSD heatmap
clean = gdf.dropna(subset=["canopy_coverage", "ESTRATO"])
tk = pairwise_tukeyhsd(clean["canopy_coverage"], clean["ESTRATO"])
diff = np.full((6, 6), np.nan)
sig = np.full((6, 6), False)
for row in tk.summary().data[1:]:
    g1, g2, d, pval, _, _, rej = row
    i, j = int(g1) - 1, int(g2) - 1
    diff[i, j] = d
    diff[j, i] = -d
    sig[i, j] = sig[j, i] = rej

sns.heatmap(diff, cmap="coolwarm", center=0, fmt=".2f",
            xticklabels=estratos, yticklabels=estratos,
            cbar_kws={"label": "Mean Difference (%)"}, ax=axes[1, 1])
for i in range(6):
    for j in range(6):
        if not np.isnan(diff[i, j]):
            text = f"{diff[i, j]:.2f}" + ("*" if sig[i, j] else "")
            axes[1, 1].text(j + 0.5, i + 0.5, text, ha="center", va="center",
                            fontsize=20, fontweight="bold" if sig[i, j] else "normal",
                            color="black" if sig[i, j] else "dimgray")
axes[1, 1].set_xlabel("Estrato")
axes[1, 1].set_ylabel("Estrato")

# Panel labels
def add_corner_labels(fig, axes, labels, dx=0.003, dy=0.006):
    for ax, lab in zip(axes.flat, labels):
        bbox = ax.get_position(fig)
        x, y = bbox.x0 - dx, bbox.y1 + dy
        fig.text(x, y, lab, fontsize=28, fontweight="bold", ha="right", va="bottom")

add_corner_labels(fig, axes, ["(a)", "(b)", "(c)", "(d)"])

plt.tight_layout()
plt.savefig(OUTPUT_FIG, dpi=400, bbox_inches="tight")
print(f"✅ Figure saved as '{OUTPUT_FIG}'")

# Optional: Show plot
# plt.show()
