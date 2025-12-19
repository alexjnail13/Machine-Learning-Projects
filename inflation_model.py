# inflation_model_fred.py

import datetime
import pandas as pd
import matplotlib.pyplot as plt

from pandas_datareader import data as pdr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -------------------------------
# STEP 1: Download CPI from FRED
# -------------------------------
start = datetime.datetime(1980, 1, 1)
end   = datetime.datetime.today()

# "CPIAUCSL" is the FRED series code for CPI All Urban Consumers: All Items
df = pdr.DataReader("CPIAUCSL", "fred", start, end)
df.columns = ["cpi"]  # rename

df.index.name = "date"
df.reset_index(inplace=True)

# -------------------------------
# STEP 2: Compute YoY inflation
# -------------------------------
df["inflation_yoy"] = df["cpi"].pct_change(12) * 100
df = df.dropna()

# -------------------------------
# STEP 3: Create binary target
# -------------------------------
THRESHOLD = 3.0  # % inflation cutoff
df["target"] = (df["inflation_yoy"] > THRESHOLD).astype(int)

# -------------------------------
# STEP 4: Feature engineering
# -------------------------------
df["inf_1m"] = df["inflation_yoy"].shift(1)
df["inf_3m"] = df["inflation_yoy"].shift(3)
df["inf_6m"] = df["inflation_yoy"].shift(6)
df = df.dropna()

# -------------------------------
# STEP 5: Train/test split
# -------------------------------
X = df[["inf_1m", "inf_3m", "inf_6m"]]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# -------------------------------
# STEP 6: Train classifier
# -------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -------------------------------
# STEP 7: Evaluate accuracy
# -------------------------------
preds = model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, preds))

# -------------------------------
# STEP 8: Probability predictions
# -------------------------------
probs = model.predict_proba(X_test)[:, 1]
df_res = df.iloc[-len(probs):].copy()
df_res["prob_above_threshold"] = probs

# -------------------------------
# STEP 9: Plot probability over time
# -------------------------------
plt.figure(figsize=(10, 5))
plt.plot(df_res["date"], df_res["prob_above_threshold"], marker="o")
plt.title(f"Probability inflation > {THRESHOLD}% (model output)")
plt.xlabel("Date")
plt.ylabel("Predicted Probability")
plt.grid()
plt.show()

# Latest probabilities
print(df_res[["date", "inflation_yoy", "prob_above_threshold"]].tail())
