# Setup

## Requirements
- Python 3.9+
- pip

## Install
```bash
pip install -r requirements.txt
```

## Place your data
Put these 3 files in `data/raw/`:
- SummerSD.csv
- WinterSD.csv
- CountriesSD.csv

## Train models
```bash
python scripts/train.py
```

## Predict
```bash
python scripts/predict.py --country "United States" --year 2028
```

## Run tests
```bash
pytest tests/ -v
```

## Open notebooks
```bash
jupyter notebook
```