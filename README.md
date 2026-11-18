# StrenML (ML_Sensor)

ML for early-age strength prediction.

Predict the **flexural strength of 3D-printed concrete** from its electrical resistivity, relative permittivity and average internal temperature, using one of three machine-learning models: Linear Regression, SVR, or ANN (MLP).

It is a static web app (HTML/CSS/JS, no server) and runs on GitHub Pages or by opening `index.html` directly.

## Inputs

| Input | Options / units |
|---|---|
| AI model | Linear Regression, SVR, ANN (MLP) |
| Concrete age | 1, 3, 5 or 7 days |
| Curing applied | Yes / No |
| Electrical resistivity | Ω·m, kΩ·m, Ω·cm (converted to Ω·m) |
| Relative permittivity | dimensionless |
| Average temperature | °C or °F (converted to °C) |

Numeric inputs must be positive numbers greater than zero.

## Models

The trained models live in `models/models.js` and are evaluated in the browser by `js/predict.js`.

> **The models in this repo are trained on synthetic demo data** (the app shows a banner while they are in use). To use the paper's models, put the dataset at `training/data.csv` with these columns:
>
> `age_days, curing, resistivity_ohm_m, permittivity, temperature_c, flexural_strength_mpa`
>
> then run:
>
> ```bash
> pip install numpy pandas scikit-learn
> python training/train_and_export.py
> ```
>
> This retrains all three models, prints test R² / MAE, and overwrites `models/models.js`.

## Deploying to GitHub Pages

Repository → Settings → Pages → Source: *Deploy from a branch*, branch `main`, folder `/ (root)`.
