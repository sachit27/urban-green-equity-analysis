# Not Your Mean Green: Urban Green Equity Analysis for Bogotá & Medellín

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Reproducible code for the paper:  
**Martinez, J., Argota Sánchez-Vaquerizo, J., & Mahajan, S. (2026).**  
*Not Your Mean Green: Beyond Averages in Mapping Socio-Spatial Inequities in Urban Greenery for Smart Cities.*  
*EPJ Data Science (Revison).*

---

## Overview

This repository provides a fully reproducible geospatial pipeline to assess socio-spatial inequities in urban greenery across socioeconomic strata (*estratos*) in **Bogotá** and **Medellín**, Colombia. The analysis integrates:

- High-resolution canopy height models (Meta/WRI, 1 m)
- Gridded population estimates (GHSL)
- Official socioeconomic stratification boundaries
- OpenStreetMap public green spaces

Unlike citywide averages used in smart city indices, our approach operates at the **residential block level** and weights results by population.

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
