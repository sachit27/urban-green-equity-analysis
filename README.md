# Urban Green Equity Analysis for Bogotá & Medellín

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Reproducible code for the paper:  
**Martinez, J., Argota Sánchez-Vaquerizo, J., & Mahajan, S. (2026).**  
*Not Your Mean Green: Beyond Averages in Mapping Socio-Spatial Inequities in Urban Greenery for Smart Cities.*  
*EPJ Data Science.*

---

## Overview

This repository provides a fully reproducible geospatial pipeline to assess socio-spatial inequities in urban greenery across socioeconomic strata (*estratos*) in **Bogotá** and **Medellín**, Colombia. The analysis integrates:

- High-resolution canopy height models (Meta/WRI, 1 m)
- Gridded population estimates (GHSL)
- Official socioeconomic stratification boundaries
- OpenStreetMap public green spaces

Unlike citywide averages used in smart city indices, our approach operates at the **residential block level** and weights results by population.

---

## Published Paper

This work has been published in EPJ Data Science:

**Martinez, J., Argota Sánchez-Vaquerizo, J., & Mahajan, S. (2026).**  
*Not Your Mean Green: Beyond Averages in Mapping Socio-Spatial Inequities in Urban Greenery for Smart Cities.*  
**EPJ Data Science, 15**, 47.  
https://link.springer.com/article/10.1140/epjds/s13688-026-00627-4

---

## Abstract

As data-driven "smart city" agendas expand across Latin America, most urban performance metrics remain focused on infrastructure, connectivity, and aggregate efficiency, often neglecting who truly benefits. Urban greenery, a vital determinant of health and climate resilience, is one such blind spot. While some frameworks now consider "green space," they do so at a coarse, citywide scale, overlooking how access is distributed across neighborhoods and social groups. This obscures critical equity gaps, particularly in cities marked by deep socio-spatial segregation. 

In this study, we develop a fully reproducible geospatial pipeline that integrates high-resolution canopy height models, public park data, gridded population estimates, and socioeconomic strata to assess how greenery is distributed, not just how much exists. Applied to Bogotá and Medellín, the method reveals stark disparities: population-weighted canopy coverage rises significantly between the lowest and highest strata, while access to public parks also shows measurable inequality, especially in high-density, underserved neighborhoods. These inequities persist despite progressive greening policies, revealing the limits of optimization when legacy segregation is ignored. Our open-source pipeline enables finer-grained, justice-oriented audits that go beyond averages to identify where greenery and its benefits are most lacking. By enabling fine-grained equity assessments, this approach underscores the importance of greenery distribution, not just quantity, as a critical indicator for inclusive and equitable smart cities.

---

## Methodology

Our analysis employs a multi-step geospatial pipeline to assess green equity at the residential block level:

### Data Integration
- **Canopy Height Models**: 1m resolution data from Meta/WRI to identify tree canopy coverage
- **Population Data**: Gridded population estimates from GHSL for population-weighted metrics
- **Socioeconomic Strata**: Official *estrato* boundaries (1-6 scale used in Colombia)
- **Public Green Spaces**: OpenStreetMap data for parks, gardens, and recreational areas

### Key Metrics
1. **Population-Weighted Canopy Coverage**: Percentage of tree canopy per residential block, weighted by population density
2. **Canopy Concentration Index (CCI)**: Measures inequality in canopy distribution across socioeconomic groups
3. **Gini Coefficient & Theil Index**: Quantify inequality in both canopy coverage and park access distances
4. **Park Access Distance**: Euclidean distance from each block centroid to nearest public park

### Analytical Approach
- Stratified analysis by socioeconomic *estrato* (1-6)
- Sensitivity analysis across multiple canopy height thresholds (1.5m, 2.0m, 3.0m)
- Spatial autocorrelation analysis using Moran's I
- Decomposition of inequality into between-group and within-group components (Theil decomposition)
- Statistical testing (ANOVA, Tukey HSD, Kruskal-Wallis) to validate differences across strata

### Key Figures from the Paper

#### Figure 2: Sensitivity Analysis
Population-weighted mean canopy coverage by *estrato* at three canopy-height thresholds (1.5, 2.0, 3.0 m) for Bogotá and Medellín. Lower thresholds yield higher absolute canopy, but the relative gradients across strata remain stable, confirming robustness of findings.
![Figure 2](https://media.springernature.com/lw685/springer-static/image/art%3A10.1140%2Fepjds%2Fs13688-026-00627-4/MediaObjects/13688_2026_627_Fig2_HTML.jpg)

#### Figure 3: Bogotá Canopy Distribution
Distribution of tree canopy cover by socioeconomic stratum in Bogotá showing:
- (a) Boxplots revealing extreme compression in lower strata vs. wide distribution in higher strata
- (b) Mean canopy coverage with 95% confidence intervals (8-fold difference between stratum 1 and 6)
- (c) Stacked bars showing categorical segregation (>99% low canopy in strata 1-3)
- (d) Heatmap of pairwise differences from Tukey HSD tests

![Figure 3](https://media.springernature.com/lw685/springer-static/image/art%3A10.1140%2Fepjds%2Fs13688-026-00627-4/MediaObjects/13688_2026_627_Fig3_HTML.jpg)

#### Figure 4: Bogotá Concentration Curve
Concentration curve for population-weighted tree-canopy coverage in Bogotá. The large shaded area between the observed distribution and the line of perfect equity yields a Canopy Concentration Index (CCI) of 0.436, indicating severe inequality.

![Figure 4](https://media.springernature.com/lw685/springer-static/image/art%3A10.1140%2Fepjds%2Fs13688-026-00627-4/MediaObjects/13688_2026_627_Fig4_HTML.jpg)

#### Figure 5: Medellín Canopy Distribution
Distribution of tree-canopy cover by socioeconomic stratum in Medellín showing more moderate inequality:
- All strata have non-zero medians with substantial overlap
- Shallowly U-shaped pattern with stratum 6 highest (14.43%) and middle strata around 5%
- Much smaller effect size compared to Bogotá

![Figure 5](https://media.springernature.com/lw685/springer-static/image/art%3A10.1140%2Fepjds%2Fs13688-026-00627-4/MediaObjects/13688_2026_627_Fig5_HTML.jpg)

#### Figure 6: Medellín Concentration Curve
Concentration curve for population-weighted tree-canopy coverage in Medellín. The curve lies much closer to the equality line, yielding a CCI of only 0.134, demonstrating markedly lower inequality than Bogotá.

![Figure 6](https://media.springernature.com/lw685/springer-static/image/art%3A10.1140%2Fepjds%2Fs13688-026-00627-4/MediaObjects/13688_2026_627_Fig6_HTML.jpg)

### Key Findings

**Bogotá**: Extreme environmental stratification
- Stratum 6 has 8× more canopy than stratum 1 (6.36% vs 0.83%)
- 26.3% of total inequality occurs between socioeconomic groups
- Strong spatial clustering (Moran's I = 0.361)

**Medellín**: Moderated inequality with complex patterns
- More equitable distribution with all strata having meaningful canopy coverage
- Only 4.9% of inequality occurs between groups
- Weaker spatial autocorrelation (Moran's I = 0.176)

**Public Park Access**: More equitable than canopy but still unequal
- Gini coefficients for park distance are lower than for canopy coverage
- Qualitative disparities (maintenance, safety) not captured by distance metrics alone

---

## How to Reproduce

### 1. Install dependencies
```bash
pip install -r requirements.txt
```
### 2. Run analyses
```bash
python bogota_green_equity.py
python medellin_green_equity.py
```
