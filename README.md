# Inflation Probability Prediction Model

This project is a **Kalshi-style financial prediction model** that estimates the probability of monthly CPI inflation exceeding a specified threshold (default: 3%). It uses **historical economic data** from the U.S. Federal Reserve (FRED) and applies **machine learning techniques** to make probabilistic forecasts.

## Features

- Loads historical CPI data from a CSV file (`cpi.csv`)
- Calculates Year-over-Year (YoY) inflation percentages
- Creates a binary target for “inflation above threshold”
- Generates lag features (past 1, 3, and 6 months) to capture trends
- Trains a **Logistic Regression** classifier
- Outputs:
  - Model accuracy on a test set
  - Probabilities of inflation exceeding the threshold for recent months
- Easy to extend with more economic indicators or advanced models

## Usage

1. Place `inflation_model.py` and `cpi.csv` in the same folder.
2. Run the script:
   ```bash
   python inflation_model.py
