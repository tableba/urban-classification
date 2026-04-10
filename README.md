
# LULC Classification of Mexico Using CNN

------------------------------------------------------------------------

## Dataset

The dataset includes:

- Sentinel-2 imagery (Bands: 2, 3, 4, 5, 6, 7, 8, 8a, 11)
- Dynamic World class labels
- Tile-based with random sampling
- 363 images

------------------------------------------------------------------------

## Project Structure

    urban-classification/
    ├── dataset/        # Data loading, preprocessing, transformations
    ├── src/            # Models, training, evaluation, visualization
    ├── output/         # Predictions, plots, metrics
    ├── main.py         # Main entry point
    ├── requirements.txt
    └── README.md

------------------------------------------------------------------------

## Installation

Install dependencies using:

    pip install -r requirements.txt

------------------------------------------------------------------------

## Usage

Run the main pipeline:

    python3 main.py

Run individual modules:

    python3 -m src.package.subpackage

