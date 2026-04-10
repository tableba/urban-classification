# LULC classification of Mexico using CNN

---

## Dataset
The dataset is generated using Google Earth Engine and includes:

- Sentinel-2 imagery (Bands: 2, 3, 4, 5, 6, 7, 8, 8a, 11)
- Dynamic World class labels
- Tile-based with random sampling
- 363 images

---

## Repository Structure
urban-classification/
├── dataset/ # data loading & preprocessing
├── src/ # models, evaluation, visualization
├── output/ # figures, predictions, graphs
├── main.py # entry point
├── requirements.txt
└── README.md

## Instructions
```
pip install -r requirements.txt
```

## Run
```
pyhton3 main.py
```

For running individual packages
```
pyhton3 -m src.package.subpackage
```
