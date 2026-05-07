# NBA 3-Point Percentage Prediction (2025–26)

**Author:** Allen Solomon
**Date:** April 16, 2026

## Overview

This project predicts NBA player 3-point percentage for the 2025–26 season using historical player statistics and shot profile data.

The model incorporates shooting volume, shot frequency, and zone-based efficiency to improve predictions.

## Data

* Source: NBA Stats API (`nba_api`)
* Seasons: 2014–15 to 2024–25
* No 2025–26 data used (to avoid leakage)

## Approach

* Feature engineering using shooting volume and shot location data
* Ridge Regression model
* Target: next-season 3P%

## Results

* Baseline RMSE: ~0.1206
* Final RMSE: ~0.1178

## Files

* `notebooks/analysis.ipynb` – full analysis and visuals
* `Scripts/` – data pipeline and modeling code
* `data/processed/predictions_2025_26.csv` – final predictions

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib nba_api

python build_core_dataset.py
python build_zone_dataset.py
python build_shot_profile_dataset.py
python merge_dataset.py
python create_training_table.py
python train_model.py
python predict_2025_26.py
```

## Notes

* API calls may take time due to rate limits
* Intermediate datasets are saved automatically

## Author

Allen Solomon
