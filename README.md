import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import glob
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
np.random.seed(42)
tf.random.set_seed(42)

try:
    import kagglehub
    download_path = kagglehub.dataset_download("sudalairajkumar/cryptocurrencypricehistory")
    candidates = glob.glob(os.path.join(download_path, "*[Bb]itcoin*.csv"))
    if candidates:
        FILE_PATH = candidates[0]
    else:
        candidates = glob.glob(os.path.join(download_path, "*.csv"))
        FILE_PATH = candidates[0] if candidates else "bitcoin.csv"
except Exception as e:
    print(f"Could not auto-detect kaggle path: {e}. Falling back to default.")
    FILE_PATH = "bitcoin.csv"

print("Loading dataset from:", FILE_PATH)
df = pd.read_csv(FILE_PATH)
print("Dataset shape:", df.shape)
print("Available columns:", df.columns.tolist())
print(df.head())

date_candidates = [
    "Date", "date", "Timestamp", "timestamp",
    "Open time", "time"
]

close_candidates = [
    "Close", "close", "Closing Price", "closing_price",
    "btc_market_price", "btc_close_usd", "BTC_USD"
]

date_col = next(
    (c for c in date_candidates if c in df.columns), None
)
close_col = next(
    (c for c in close_candidates if c in df.columns), None
)

if date_col is None or close_col is None:
    raise ValueError(
        "Date/Close column not found. Set date_col and "
        "close_col manually using your CSV column names."
    )
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df[close_col] = pd.to_numeric(df[close_col], errors="coerce")
df = df.dropna(subset=[date_col])
df = df.sort_values(date_col)
df = df[[date_col, close_col]].copy()
df = df.rename(columns={
    date_col: "Date",
    close_col: "Close"
})
df["Date"] = df["Date"].dt.normalize()
df = df.groupby("Date", as_index=False)["Close"].mean()
df = df.sort_values("Date").set_index("Date")
daily_index = pd.date_range(
    start=df.index.min(),
    end=df.index.max(),
    freq="D"
)
df = df.reindex(daily_index)
df.index.name = "Date"
df.loc[df["Close"] <= 0, "Close"] = np.nan
df["Close"] = df["Close"].interpolate(method="linear")
df["Close"] = df["Close"].ffill().bfill()
df = df.dropna(subset=["Close"])
df = df.tail(2000).copy()
if len(df) < 100:
    raise ValueError(
        "Too few valid daily records. Check the dataset."
    )
print("\nCleaned dataset:", df.shape)
print("Date range:", df.index.min(), "to", df.index.max())
print("Remaining missing values:", df["Close"].isna().sum())
prices = df[["Close"]].values.astype("float32")
split_index = int(len(prices) * 0.80)
if split_index <= 7 or split_index >= len(prices):
    raise ValueError("Not enough records for train/test split.")
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(prices[:split_index])
scaled_prices = scaler.transform(prices)
WINDOW_SIZE = 7
X, y, target_indices = [], [], []
for i in range(WINDOW_SIZE, len(scaled_prices)):
    X.append(scaled_prices[i - WINDOW_SIZE:i, 0])
    y.append(scaled_prices[i, 0])
    target_indices.append(i)
X = np.array(X, dtype="float32").reshape(-1, WINDOW_SIZE, 1)
y = np.array(y, dtype="float32").reshape(-1, 1)
target_indices = np.array(target_indices)
train_mask = target_indices < split_index
test_mask = target_indices >= split_index
X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]
test_dates = df.index[target_indices[test_mask]]
print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
print("Input shape:", X_train.shape)
model = Sequential([
    Input(shape=(WINDOW_SIZE, 1)),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25, activation="relu"),
    Dense(1)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="mean_squared_error"
)
model.summary()
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    shuffle=False,
    callbacks=[early_stopping],
    verbose=1
)
pred_scaled = model.predict(X_test, verbose=0)
y_test_actual = scaler.inverse_transform(y_test).ravel()
pred_actual = scaler.inverse_transform(pred_scaled).ravel()
mae = mean_absolute_error(y_test_actual, pred_actual)
rmse = np.sqrt(mean_squared_error(y_test_actual, pred_actual))

print("\n========== TEST RESULTS ==========")
print(f"MAE:  {mae:,.2f}")
print(f"RMSE: {rmse:,.2f}")
print("Currency units match the dataset's Close column.")
previous_close = prices[target_indices[test_mask] - 1, 0]
baseline_mae = mean_absolute_error(
    y_test_actual, previous_close
)
baseline_rmse = np.sqrt(mean_squared_error(
    y_test_actual, previous_close
))
print("\n========== LAST-CLOSE BASELINE ==========")
print(f"Baseline MAE:  {baseline_mae:,.2f}")
print(f"Baseline RMSE: {baseline_rmse:,.2f}")
plt.figure(figsize=(14, 6))
plt.plot(
    test_dates, y_test_actual,
    label="Actual Closing Price", color="blue"
)
plt.plot(
    test_dates, pred_actual,
    label="LSTM Predicted Price",
    color="red", alpha=0.8
)
plt.title("Bitcoin: Actual vs Predicted Closing Prices")
plt.xlabel("Date")
plt.ylabel("Closing Price")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=300)
plt.show()
plt.figure(figsize=(10, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("LSTM Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error (scaled prices)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("training_loss.png", dpi=300)
plt.show()
results = pd.DataFrame({
    "Date": test_dates,
    "Actual_Close": y_test_actual,
    "Predicted_Close": pred_actual
})
results.to_csv("test_predictions.csv", index=False)
last_seven_days = scaled_prices[-WINDOW_SIZE:].reshape(
    1, WINDOW_SIZE, 1
)
next_scaled = model.predict(last_seven_days, verbose=0)
next_close = scaler.inverse_transform(next_scaled)[0, 0]
next_date = df.index[-1] + pd.Timedelta(days=1)
print("\n========== NEXT-DAY FORECAST ==========")
print("Last available date:", df.index[-1].date())
print("Forecast date:", next_date.date())
print(f"Predicted next-day Close: {next_close:,.2f}")
model.save("bitcoin_lstm.keras")
print("\nSaved: bitcoin_lstm.keras")
print("Saved: actual_vs_predicted.png")
print("Saved: training_loss.png")
print("Saved: test_predictions.csv")

# =========================================================================
# GITHUB DISPATCH INTEGRATION
# =========================================================================

print("\n========== PREPARING GITHUB UPLOAD ==========")
try:
    from google.colab import userdata
    github_token = userdata.get('GITHUB_TOKEN')
except Exception as e:
    github_token = None
    print("Warning: GITHUB_TOKEN Secret not found. Please add your token under Secrets (Key icon).")

if github_token:
    username = "KaviyanV"
    email = "kavi052007@gmail.com"
    repo_name = "bitcoin-lstm-forecast"

    # Generate a script-only Python file to store the core training logic in the repo
    with open("bitcoin_forecast.py", "w") as f_out:
        # Write current logic into a file without the GitHub bootstrap code itself
        f_out.write("# Auto-generated code from Colab\n")
        f_out.write("\n".join(open(__file__).readlines()[:-50]))

    # Shell operations to initialize git and push
    os.system(f'git config --global user.name "{username}"')
    os.system(f'git config --global user.email "{email}"')
    os.system('git init -b main')
    os.system('git add bitcoin_forecast.py actual_vs_predicted.png training_loss.png test_predictions.csv')
    os.system('git commit -m "Initial commit: Bitcoin LSTM forecast model and outputs"')

    # Attempt to create the GitHub repository via API
    import requests
    headers = {"Authorization": f"token {github_token}"}
    req = requests.post("https://api.github.com/user/repos", json={"name": repo_name, "private": False}, headers=headers)
    if req.status_code in [201, 422]:
        print(f"Repository target '{repo_name}' initialized successfully.")
        os.system(f'git remote add origin https://{github_token}@github.com/{username}/{repo_name}.git')
        os.system('git branch -M main')
        push_status = os.system('git push -u origin main --force')
        if push_status == 0:
            print(f"\nSuccess! Code pushed to: https://github.com/{username}/{repo_name}")
        else:
            print("\nPush failed. Check write access configurations.")
    else:
        print(f"Could not create GitHub repo. Response: {req.json()}")
else:
    print("Skipping GitHub push: No GITHUB_TOKEN Secret detected.")
