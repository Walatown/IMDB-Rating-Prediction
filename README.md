# IMDb Rating Prediction – README

This project is centered on the Jupyter notebook **`imdb_final.ipynb`**, which explores, cleans, and models a Kaggle movie dataset to understand what drives IMDb user ratings and to build baseline predictive models.

## 1) What you need to download

Download the following dataset and place it in the project folder as shown below:

- **`movies.csv`** — from Kaggle dataset **“Movie Industry” by Daniel Grijalva**  
  Source: https://www.kaggle.com/datasets/danielgrijalvas/movies  
  > The notebook expects a local file named **`movies.csv`** (no subfolder path is hard‑coded).

## 2) Environment & dependencies

The notebook uses Python and the libraries below (derived from imports in the notebook). Create and activate a virtual environment, then install these packages:


> **Notes**
> - `shap` may require compatible versions of `numba`/`numpy` depending on your Python version. If you hit build issues, try: `pip install shap==0.42.1` and ensure `numba` < 0.60 on older Pythons.
> - `plotly` is used in offline mode within the notebook.
> - The notebook imports `scipy.io.arff`, but the workflow is based on `movies.csv`; no ARFF file is required.

## 3) How to run

1. **Set up env (example with `venv`):**
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # On Windows: .venv\\Scripts\\activate
   pip install --upgrade pip
   pip install jupyter pandas numpy scikit-learn scipy matplotlib seaborn plotly shap squarify
Put the data file in place:

Ensure movies.csv (from the Kaggle link above) is in the same folder as imdb_final.ipynb.

Launch Jupyter and open the notebook:

Always show details
jupyter notebook

## 4) What the notebook does (at a glance)

Loads movies.csv (6,820 movies; ~1986–2016) with columns like name, genre, budget, gross, released, runtime, score (IMDb user rating), votes, star, writer, company, country, etc.

Performs exploratory data analysis (EDA) and visualizations (Matplotlib/Seaborn/Plotly/Squarify).

Preprocesses features and splits data.

Trains baseline ML models (e.g., LinearRegression, KNeighbors, SVR) using scikit‑learn.

Uses SHAP to interpret feature importance for model predictions (if enabled).

## 5) Troubleshooting

FileNotFoundError: 'movies.csv' → Place the Kaggle file next to the notebook or update the path in the pd.read_csv(...) call.

shap/numba install errors → Pin versions (e.g., pip install shap==0.42.1 numba<0.60) or upgrade pip/build tools.

Plotly not showing → In Jupyter, outputs appear inline; ensure you’re running all cells and not in a headless environment.

## 6) Citation

Dataset:
Grijalva, D. (2020). Movie Industry. Kaggle. https://www.kaggle.com/datasets/danielgrijalvas/movies
