# Rental Price Prediction — Ho Chi Minh City

Predicting rental room prices in HCMC from multi-source scraped data.

## Project Structure

rental-price-prediction-hcmc/
├── data/
│ ├── raw/
│ ├── processed/
│ └── external/
├── notebooks/
│ ├── scraping/
│ ├── cleaning/
│ ├── eda/
│ ├── feature_engineering/
│ └── modeling/
├── src/
│ ├── scraping/
│ ├── cleaning/
│ ├── feature_engineering/
│ ├── modeling/
│ └── utils/
└── reports/
├── figures/
├── slides/
└── final_report/

## Quickstart

- Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

- Install dependencies:

```powershell
pip install -r requirements.txt
```

## Data Layout

- Place raw scraped files in `data/raw/`.
- Save cleaned, merged datasets in `data/processed/`.
- Put third-party/reference files in `data/external/`.

## Notes

- Keep large data out of Git history (see `.gitignore`).
- Notebooks are organized by pipeline stage.
