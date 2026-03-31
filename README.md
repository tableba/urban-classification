# Urban detection across mexico between 2016 and 2023

## Overview
This repository contains the code and resources for the thesis project:

**"Title of Thesis"**

The project focuses on using machine learning to detect urban areas from satellite imagery and quantify urban expansion over time.

---

## Objectives
- Train supervised machine-learning models to classify pixels as **urban** or **non-urban**
- Apply models to multi-temporal satellite imagery (2016 vs 2023)
- Quantify urban growth between the two time periods
- Evaluate model performance using building-based ground-truth data

---

## Dataset
The dataset is generated using Google Earth Engine and includes:

- Sentinel-2 imagery (RGB + NDVI)
- Time periods: **2016 and 2023**
- Building masks:
  - 2016 buildings
  - 2023 buildings
  - New buildings (2016 → 2023)
- Tile-based structure with aligned spatial bounds

> NOTE: Dataset is not included in this repository. See instructions below.

---

## Repository Structure
