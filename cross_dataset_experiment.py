"""
Cross-Dataset Experiment: 使用原論文的時間區間重現實驗
原論文: Training 2009-01 ~ 2018-12, Trading 2019-01 ~ 2020-07
"""

from __future__ import annotations
import itertools
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# Part 0. 設定原論文時間區間
# ============================================================
TRAIN_START = "2009-01-01"
TRAIN_END = "2018-12-31"
TRADE_START = "2019-01-01"
TRADE_END = "2020-07-01"

print("=" * 80)
print("Cross-Dataset Experiment — 使用原論文時間區間")
print(f"Training: {TRAIN_START} ~ {TRAIN_END}")
print(f"Trading:  {TRADE_START} ~ {TRADE_END}")
print("=" * 80)

# ============================================================
# Part 1. 下載與預處理數據
# ============================================================
from finrl import config_tickers
from finrl.config import INDICATORS
from finrl.meta.preprocessor.preprocessors import data_split, FeatureEngineer
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader

# 使用原論文的 DOW 30（2019 年版本）
DOW_30_2019 = [
    "AXP", "AMGN", "AAPL", "BA", "CAT", "CSCO", "CVX", "GS", "HD",
    "HON", "IBM", "INTC", "JNJ", "KO", "JPM", "MCD", "MMM", "MRK",
    "MSFT", "NKE", "PG", "TRV", "UNH", "CRM", "VZ", "V", "WBA",
    "WMT", "DIS", "DOW",
]

print("\n[Step 1/4] 下載 DOW 30 數據...")
df_raw = YahooDownloader(
    start_date=TRAIN_START,
    end_date=TRADE_END,
    ticker_list=DOW_30_2019,
).fetch_data()
print(f"  Raw data shape: {df_raw.shape}")

print("[Step 1/4] 特徵工程...")
fe = FeatureEngineer(
    use_technical_indicator=True,
    tech_indicator_list=INDICATORS,
    use_vix=True,
    use_turbulence=True,
    user_defined_feature=False,
)
processed = fe.preprocess_data(df_raw)

list_ticker = processed["tic"].unique().tolist()
list_date = list(pd.date_range(processed["date"].min(), processed["date"].max()).astype(str))
combination = list(itertools.product(list_date, list_ticker))

processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(
    processed, on=["date", "tic"], how="left"
)
processed_full = processed_full[processed_full["date"].isin(processed["date"])]
processed_full = processed_full.sort_values(["date", "tic"])
processed_full = processed_full.fillna(0)

train = data_split(processed_full, TRAIN_START, TRAIN_END)
trade = data_split(processed_full, TRADE_START, TRADE_END)
print(f"  Train: {len(train)} rows, Trade: {len(trade)} rows")

# ============================================================
# Part 2. 建立環境 & 訓練
# ============================================================
from stable_baselines3.common.logger import configure
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

CROSS_MODEL_DIR = "trained_models_cross"
CROSS_RESULTS_DIR = "results_cross"
os.makedirs(CROSS_MODEL_DIR, exist_ok=True)
os.makedirs(CROSS_RESULTS_DIR, exist_ok=True)

stock_dimension = len(train.tic.unique())
state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
print(f"\n[Step 2/4] Stock Dim: {stock_dimension}, State Space: {state_space}")

buy_cost_list = sell_cost_list = [0.001] * stock_dimension
num_stock_shares = [0] * stock_dimension

env_kwargs = {
    "hmax": 100,
    "initial_amount": 1000000,
    "num_stock_shares": num_stock_shares,
    "buy_cost_pct": buy_cost_list,
    "sell_cost_pct": sell_cost_list,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4,
}

e_train_gym = StockTradingEnv(df=train, **env_kwargs)
env_train, _ = e_train_gym.get_sb_env()

# 使用較多訓練步數（50000）以獲得更好的結果
TOTAL_TIMESTEPS = 50000

agents_config = {
    "a2c":  {},
    "ddpg": {},
    "ppo":  {"n_steps": 2048, "ent_coef": 0.01, "learning_rate": 0.00025, "batch_size": 128},
    "td3":  {"batch_size": 100, "buffer_size": 1000000, "learning_rate": 0.001},
    "sac":  {"batch_size": 128, "buffer_size": 100000, "learning_rate": 0.0001,
             "learning_starts": 100, "ent_coef": "auto_0.1"},
}

trained_models = {}
for name, kwargs in agents_config.items():
    print(f"\n[Step 2/4] Training {name.upper()} ({TOTAL_TIMESTEPS} steps)...")
    agent = DRLAgent(env=env_train)
    model = agent.get_model(name, model_kwargs=kwargs if kwargs else {})
    tmp_path = CROSS_RESULTS_DIR + f"/{name}"
    new_logger = configure(tmp_path, ["csv"])
    model.set_logger(new_logger)
    trained = agent.train_model(model=model, tb_log_name=name, total_timesteps=TOTAL_TIMESTEPS)
    trained.save(CROSS_MODEL_DIR + f"/agent_{name}")
    trained_models[name] = trained
    print(f"  {name.upper()} trained and saved.")

# ============================================================
# Part 3. 回測
# ============================================================
print("\n[Step 3/4] Backtesting...")

e_trade_gym = StockTradingEnv(
    df=trade, turbulence_threshold=70, risk_indicator_col="vix", **env_kwargs
)

results = {}
for name, model in trained_models.items():
    df_av, df_actions = DRLAgent.DRL_prediction(model=model, environment=e_trade_gym)
    results[name] = df_av
    print(f"  {name.upper()}: Final = ${df_av['account_value'].iloc[-1]:,.0f}")

# DJIA baseline
import yfinance as yf
df_dji = yf.download("^DJI", start=TRADE_START, end=TRADE_END)
df_dji = df_dji[["Close"]].reset_index()
df_dji.columns = ["date", "close"]
df_dji["date"] = df_dji["date"].astype(str)
fst_day = df_dji["close"].iloc[0]
dji_values = df_dji["close"].div(fst_day).mul(1000000).values.flatten()

# ============================================================
# Part 4. 計算指標
# ============================================================
print("\n[Step 4/4] 計算指標...")

def calc_metrics(values, name, trading_days=252):
    daily_ret = np.diff(values) / values[:-1]
    total_ret = (values[-1] - values[0]) / values[0] * 100
    n = len(values)
    annual_ret = ((values[-1] / values[0]) ** (trading_days / n) - 1) * 100
    sharpe = np.sqrt(trading_days) * np.mean(daily_ret) / (np.std(daily_ret) + 1e-9)
    peak = np.maximum.accumulate(values)
    dd = (peak - values) / peak
    mdd = np.max(dd) * 100
    vol = np.std(daily_ret) * np.sqrt(trading_days) * 100
    return {
        "name": name, "final": values[-1], "total_return": total_ret,
        "annual_return": annual_ret, "sharpe": sharpe, "mdd": mdd, "vol": vol
    }

print("\n" + "=" * 100)
print(f"{'Strategy':6s} | {'Final Value':>14s} | {'Total Ret':>10s} | {'Annual Ret':>10s} | {'Sharpe':>8s} | {'MDD':>8s} | {'Vol':>8s}")
print("=" * 100)

all_metrics = []
for name, df_av in results.items():
    m = calc_metrics(df_av["account_value"].values, name.upper())
    all_metrics.append(m)
    print(f"{m['name']:6s} | ${m['final']:>12,.0f} | {m['total_return']:>+9.2f}% | {m['annual_return']:>+9.2f}% | {m['sharpe']:>8.4f} | {m['mdd']:>7.2f}% | {m['vol']:>7.2f}%")

m = calc_metrics(dji_values, "DJIA")
all_metrics.append(m)
print(f"{m['name']:6s} | ${m['final']:>12,.0f} | {m['total_return']:>+9.2f}% | {m['annual_return']:>+9.2f}% | {m['sharpe']:>8.4f} | {m['mdd']:>7.2f}% | {m['vol']:>7.2f}%")
print("=" * 100)

# 原論文指標
print("\n--- 原論文報告指標 (Ensemble, 2019/01 ~ 2020/07) ---")
print(f"{'Ensemble':6s} |                | {'':>10s} | {'+19.19':>9s}% | {'1.3000':>8s} | {'11.52':>7s}% |")
print(f"{'DJIA':6s} |                | {'':>10s} | {'+14.97':>9s}% | {'0.9600':>8s} | {'33.95':>7s}% |")

# ============================================================
# Part 5. 繪圖
# ============================================================
plt.rcParams["figure.figsize"] = (15, 6)
plt.rcParams["font.size"] = 12
fig, ax = plt.subplots()

colors = {"a2c": "#2196F3", "ddpg": "#FF9800", "ppo": "#4CAF50", "td3": "#F44336", "sac": "#9C27B0"}
for name, df_av in results.items():
    dates = pd.to_datetime(df_av.iloc[:, 0])
    ax.plot(dates, df_av["account_value"].values, label=name.upper(), color=colors[name], linewidth=1.5)

# DJIA
dji_dates = pd.to_datetime(df_dji["date"])
ax.plot(dji_dates, dji_values, label="DJIA", color="#E91E63", linewidth=2, linestyle="--")

# 原論文 Ensemble 參考線
ax.axhline(y=1191900, color="gold", linewidth=1.5, linestyle=":", label="Paper Ensemble (+19.19%)")

ax.set_title("Cross-Dataset Experiment: Trading Period 2019/01 ~ 2020/07\n(Original Paper Time Period)", fontsize=14)
ax.set_xlabel("Date")
ax.set_ylabel("Portfolio Value ($)")
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("cross_dataset_result.png", dpi=150, bbox_inches="tight")
print("\nPlot saved to cross_dataset_result.png")
print("\n✅ Cross-dataset experiment complete!")
