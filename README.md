# Inflation Probability Prediction Model

This project is a **Kalshi-style financial prediction model** that estimates the probability of monthly CPI inflation exceeding a specified threshold (default: 3%). It uses **real historical economic data from the U.S. Federal Reserve (FRED)** and applies **machine learning** techniques to make probabilistic forecasts.

---

## Features

- Automatically downloads CPI data from FRED (no CSV file needed)
- Computes **Year-over-Year (YoY) inflation** percentages
- Creates a **binary target** for “inflation above threshold”
- Generates **lag features** (1, 3, and 6 months) to capture trends
- Trains a **Logistic Regression** classifier
- Outputs:
  - Model accuracy on a test set
  - Probabilities of inflation exceeding the threshold for recent months
- Plots the **probability trend over time**  
- Easy to extend with additional economic indicators or more advanced models

---

## Requirements

- Python 3.x
- Packages:
  - `pandas`
  - `scikit-learn`
  - `pandas-datareader`
  - `matplotlib`

Install packages via pip:

```bash
pip install pandas scikit-learn pandas-datareader matplotlib
