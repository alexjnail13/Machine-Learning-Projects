# inflation_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -------------------------------
# STEP 1: Load data
# -------------------------------
df = pd.read_csv("cpi.csv")  # put cpi.csv in the same folder as this script
df.columns = ["date", "cpi"]  # rename for clarity
df["date"] = pd.to_datetime(df["date"])

# -------------------------------
# STEP 2: Create inflation % (YoY)
# -------------------------------
df["inflation_yoy"] = df["cpi"].pct_change(12) * 100
df = df.dropna()

# -------------------------------
# STEP 3: Create target label
# -------------------------------
THRESHOLD = 3.0  # change if you want a different threshold
df["target"] = (df["inflation_yoy"] > THRESHOLD).astype(int)

# -------------------------------
# STEP 4: Feature engineering
# -------------------------------
# Use past months’ inflation as features
df["inflation_1m"] = df["inflation_yoy"].shift(1)
df["inflation_3m"] = df["inflation_yoy"].shift(3)
df["inflation_6m"] = df["inflation_yoy"].shift(6)
df = df.dropna()

# -------------------------------
# STEP 5: Prepare data for ML
# -------------------------------
X = df[["inflation_1m", "inflation_3m", "inflation_6m"]]
y = df["target"]

# Split into train/test (keep time order)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# -------------------------------
# STEP 6: Train model
# -------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -------------------------------
# STEP 7: Evaluate model
# -------------------------------
preds = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))

# -------------------------------
# STEP 8: Predict probabilities
# -------------------------------
probs = model.predict_proba(X_test)[:, 1]  # probability inflation > 3%
df_results = df.iloc[-len(probs):].copy()
df_results["prob_above_3"] = probs

# Show last 10 predictions
print(df_results[["date", "inflation_yoy", "prob_above_3"]].tail(10))
