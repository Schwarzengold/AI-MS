# Energy Consumption Regression

## Overview

This project builds and evaluates simple linear regression models to forecast electricity consumption in a city district based on environmental and temporal features.

- **Task 1:** Linear regression on **numeric** features  
  `temperature`, `humidity`, `hour`, `is_weekend`  
- **Task 2:** Extend Task 1 by adding **categorical** features  
  `season`, `district_type` (One‑Hot encoding)

## Files

- `energy_regression.py` — main script:  
  - loads & validates data  
  - trains both models  
  - prints MAPE for each task  
  - plots “True vs Predicted” scatter plots  
- `energy_usage_plus.csv` — example dataset  
- `requirements.txt` — list of Python dependencies

## Installation

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

python energy_regression.py --data energy_usage_plus.csv
