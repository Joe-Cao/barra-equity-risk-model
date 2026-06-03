"""
Barra-style Equity Risk Model Project

Inputs expected in BASE_DIR:
  - target stocks/                         # folder containing *.txt stock files
  - barra_sp500_500_expected_filenames.txt # one filename per line, e.g. aapl.us.txt
  - barra_sp500_500_gics_sectors.txt       # one GICS Sector per line, aligned with filenames

Outputs:
  - barra_outputs/factor_returns.csv
  - barra_outputs/specific_residuals.csv
  - barra_outputs/backtest_results.csv
  - barra_outputs/backtest_metrics.csv
  - barra_outputs/missing_files.txt
  - barra_outputs/plot1_predicted_vs_realized_vol.png
  - barra_outputs/plot2_standardized_return_hist.png
  - barra_outputs/plot3_risk_decomposition.png
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. User settings: change BASE_DIR to your actual folder
# ============================================================

BASE_DIR = Path(r"C:\Users\caoxi\python练习\Barra")  # <-- change this

TARGET_STOCKS_DIR = BASE_DIR / "target stocks"
EXPECTED_FILENAMES_PATH = BASE_DIR / "barra_sp500_500_expected_filenames.txt"
GICS_SECTORS_PATH = BASE_DIR / "barra_sp500_500_gics_sectors.txt"

OUTPUT_DIR = BASE_DIR / "barra_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

DATA_START_DATE = "2022-01-01"
DATA_END_DATE = "2025-12-31"

# Rolling windows
MOMENTUM_WINDOW = 252
MOMENTUM_SKIP = 21
VOL_WINDOW = 252
LIQUIDITY_WINDOW = 21
BETA_WINDOW = 252

FACTOR_COV_WINDOW = 252
SPECIFIC_RISK_WINDOW = 252
REALIZED_VOL_WINDOW = 21

MIN_REGRESSION_STOCKS = 100
MIN_FACTOR_COV_OBS = 126
MIN_SPECIFIC_OBS = 60

TRADING_DAYS_PER_YEAR = 252


# ============================================================
# 2. Helper functions
# ============================================================

def ticker_from_filename(filename: str) -> str:
    """Convert 'aapl.us.txt' -> 'AAPL'."""
    filename = filename.strip()
    if filename.lower().endswith(".us.txt"):
        return filename[:-7].upper()
    if filename.lower().endswith(".txt"):
        return filename[:-4].upper()
    return filename.upper()


def clean_factor_name(x: str) -> str:
    return (
        x.strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("&", "and")
        .replace("-", "_")
    )


def read_lines(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def read_stock_file(path: Path, ticker: str) -> pd.DataFrame:
    """
    Read one MetaStock-like daily txt file.

    Expected columns:
      <TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>
    """
    df = pd.read_csv(path)

    # Remove angle brackets and upper-case names.
    df.columns = [c.strip().replace("<", "").replace(">", "").upper() for c in df.columns]

    required = {"DATE", "CLOSE", "VOL"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")

    df["date"] = pd.to_datetime(df["DATE"].astype(str), format="%Y%m%d", errors="coerce")
    df["ticker"] = ticker
    df["close"] = pd.to_numeric(df["CLOSE"], errors="coerce")
    df["volume"] = pd.to_numeric(df["VOL"], errors="coerce")

    df = df[["date", "ticker", "close", "volume"]]
    df = df.dropna(subset=["date", "close", "volume"])
    df = df[(df["date"] >= DATA_START_DATE) & (df["date"] <= DATA_END_DATE)]
    df = df[df["close"] > 0]
    df = df[df["volume"] >= 0]

    # In rare cases there may be duplicate dates. Keep the last record.
    df = df.sort_values("date").drop_duplicates(["ticker", "date"], keep="last")
    return df


def cross_sectional_zscore(x: pd.Series) -> pd.Series:
    """Daily cross-sectional standardization."""
    x = x.replace([np.inf, -np.inf], np.nan)
    mu = x.mean(skipna=True)
    sd = x.std(skipna=True, ddof=0)
    if not np.isfinite(sd) or sd < 1e-12:
        return pd.Series(np.nan, index=x.index)
    return (x - mu) / sd


def safe_cov_matrix(df: pd.DataFrame) -> pd.DataFrame | None:
    """Return covariance matrix if enough clean observations exist."""
    clean = df.dropna(how="any")
    if len(clean) < MIN_FACTOR_COV_OBS:
        return None
    return clean.cov()


def make_exposure_matrix_for_date(
    date: pd.Timestamp,
    style_exposures: dict[str, pd.DataFrame],
    industry_dummies: pd.DataFrame,
    factor_cols: list[str],
) -> pd.DataFrame:
    """
    Build stock x factor exposure matrix X_t.
    Industry dummies are static. Style exposures are date-varying.
    """
    tickers = industry_dummies.index
    X = industry_dummies.copy()

    for name, mat in style_exposures.items():
        if date in mat.index:
            X[name] = mat.loc[date].reindex(tickers)
        else:
            X[name] = np.nan

    return X.reindex(columns=factor_cols)


# ============================================================
# 3. Load file list and GICS sectors
# ============================================================

expected_filenames = read_lines(EXPECTED_FILENAMES_PATH)
gics_sectors = read_lines(GICS_SECTORS_PATH)

if len(expected_filenames) != len(gics_sectors):
    raise ValueError(
        f"Line count mismatch: expected filenames={len(expected_filenames)}, "
        f"GICS sectors={len(gics_sectors)}"
    )

tickers = [ticker_from_filename(x) for x in expected_filenames]
sector_map = dict(zip(tickers, gics_sectors))

print(f"Target tickers: {len(tickers)}")


# ============================================================
# 4. Load stock price and volume data
# ============================================================

all_frames = []
missing_files = []

for filename, ticker in zip(expected_filenames, tickers):
    path = TARGET_STOCKS_DIR / filename

    if not path.exists():
        missing_files.append(filename)
        continue

    try:
        one = read_stock_file(path, ticker)
        if not one.empty:
            all_frames.append(one)
    except Exception as e:
        print(f"Failed to read {filename}: {e}")
        missing_files.append(filename)

missing_path = OUTPUT_DIR / "missing_files.txt"
with open(missing_path, "w", encoding="utf-8") as f:
    for x in missing_files:
        f.write(x + "\n")

if not all_frames:
    raise RuntimeError("No stock files loaded. Check TARGET_STOCKS_DIR and filenames.")

prices_long = pd.concat(all_frames, ignore_index=True)

print(f"Loaded stock files: {prices_long['ticker'].nunique()}")
print(f"Missing files: {len(missing_files)}")
print(f"Date range: {prices_long['date'].min().date()} to {prices_long['date'].max().date()}")

close = prices_long.pivot(index="date", columns="ticker", values="close").sort_index()
volume = prices_long.pivot(index="date", columns="ticker", values="volume").sort_index()

# Keep only tickers actually loaded and with sector info.
loaded_tickers = [t for t in tickers if t in close.columns and t in sector_map]
close = close[loaded_tickers]
volume = volume[loaded_tickers]

# Log returns are convenient for risk modeling.
returns = np.log(close / close.shift(1))


# ============================================================
# 5. Build style factors: Momentum, Volatility, Liquidity, Beta
# ============================================================

# Momentum: past 252 trading days, skipping the most recent 21 trading days.
momentum_raw = returns.rolling(MOMENTUM_WINDOW, min_periods=MOMENTUM_WINDOW).sum().shift(MOMENTUM_SKIP)

# Volatility: 252-day rolling standard deviation of daily log returns.
volatility_raw = returns.rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std()

# Liquidity: log of 21-day average dollar volume.
dollar_volume = close * volume
liquidity_raw = np.log(
    dollar_volume.rolling(LIQUIDITY_WINDOW, min_periods=LIQUIDITY_WINDOW).mean()
)

# Beta: 252-day rolling beta against equal-weight market return.
market_return = returns.mean(axis=1, skipna=True)
market_var = market_return.rolling(BETA_WINDOW, min_periods=BETA_WINDOW).var()
beta_raw = returns.rolling(BETA_WINDOW, min_periods=BETA_WINDOW).cov(market_return).div(market_var, axis=0)

# Cross-sectional standardization by date.
style_exposures = {
    "Momentum": momentum_raw.apply(cross_sectional_zscore, axis=1),
    "Volatility": volatility_raw.apply(cross_sectional_zscore, axis=1),
    "Liquidity": liquidity_raw.apply(cross_sectional_zscore, axis=1),
    "Beta": beta_raw.apply(cross_sectional_zscore, axis=1),
}


# ============================================================
# 6. Build industry factor dummies from GICS Sector
# ============================================================

sector_series = pd.Series({t: sector_map[t] for t in loaded_tickers}, name="sector")
industry_dummies = pd.get_dummies(sector_series)
industry_dummies.columns = [f"IND_{clean_factor_name(c)}" for c in industry_dummies.columns]
industry_dummies = industry_dummies.astype(float)

industry_factor_cols = list(industry_dummies.columns)
style_factor_cols = ["Momentum", "Volatility", "Liquidity", "Beta"]
factor_cols = industry_factor_cols + style_factor_cols

print(f"Industry factors: {len(industry_factor_cols)}")
print(f"Total factors: {len(factor_cols)}")


# ============================================================
# 7. Daily cross-sectional regressions
#
# Model:
#   r_{i,t+1} = X_{i,t} f_{t+1} + epsilon_{i,t+1}
#
# No intercept is used because all GICS sector dummies are included.
# ============================================================

dates = close.index
factor_returns = []
residuals = []

for idx in range(len(dates) - 1):
    d = dates[idx]
    next_d = dates[idx + 1]

    X = make_exposure_matrix_for_date(d, style_exposures, industry_dummies, factor_cols)
    y = returns.loc[next_d].reindex(X.index)

    reg_df = X.copy()
    reg_df["y"] = y

    reg_df = reg_df.replace([np.inf, -np.inf], np.nan).dropna(how="any")

    if len(reg_df) < max(MIN_REGRESSION_STOCKS, len(factor_cols) + 10):
        continue

    X_np = reg_df[factor_cols].to_numpy(dtype=float)
    y_np = reg_df["y"].to_numpy(dtype=float)

    try:
        coef, *_ = np.linalg.lstsq(X_np, y_np, rcond=None)
    except np.linalg.LinAlgError:
        continue

    fitted = X_np @ coef
    eps = y_np - fitted

    factor_returns.append(pd.Series(coef, index=factor_cols, name=next_d))
    residuals.append(pd.Series(eps, index=reg_df.index, name=next_d))

factor_returns_df = pd.DataFrame(factor_returns).sort_index()
residuals_df = pd.DataFrame(residuals).sort_index()

factor_returns_df.to_csv(OUTPUT_DIR / "factor_returns.csv")
residuals_df.to_csv(OUTPUT_DIR / "specific_residuals.csv")

print(f"Factor return observations: {len(factor_returns_df)}")


# ============================================================
# 8. Ex-ante risk prediction and backtest
#
# At date t:
#   Use X_t, past factor returns up to t to estimate F_t,
#   use past residuals up to t to estimate D_t,
#   predict next-day portfolio risk.
#
# Portfolio:
#   Equal-weight among all eligible stocks on date t.
# ============================================================

backtest_rows = []

for idx in range(len(dates) - 1):
    d = dates[idx]
    next_d = dates[idx + 1]

    if d not in factor_returns_df.index or d not in residuals_df.index:
        continue

    # Past factor returns up to date d.
    past_f = factor_returns_df.loc[:d].tail(FACTOR_COV_WINDOW)
    if len(past_f.dropna(how="any")) < MIN_FACTOR_COV_OBS:
        continue

    F = safe_cov_matrix(past_f)
    if F is None:
        continue

    # Past residuals up to date d.
    past_eps = residuals_df.loc[:d].tail(SPECIFIC_RISK_WINDOW)
    if len(past_eps) < MIN_SPECIFIC_OBS:
        continue

    specific_var = past_eps.var(axis=0, skipna=True, ddof=1)

    X = make_exposure_matrix_for_date(d, style_exposures, industry_dummies, factor_cols)
    y_next = returns.loc[next_d].reindex(X.index)

    # Eligible stocks must have complete exposure, next-day return and specific variance.
    eligible = (
        X.notna().all(axis=1)
        & y_next.notna()
        & specific_var.reindex(X.index).notna()
    )

    X_e = X.loc[eligible, factor_cols]
    y_e = y_next.loc[eligible]
    dvar_e = specific_var.reindex(X_e.index)

    if len(X_e) < max(MIN_REGRESSION_STOCKS, len(factor_cols) + 10):
        continue

    # Equal-weight portfolio over eligible stocks.
    n = len(X_e)
    w = pd.Series(1.0 / n, index=X_e.index)

    # Factor exposure of the portfolio:
    #   b_p = X_t' w
    b = X_e.T @ w

    F_aligned = F.reindex(index=factor_cols, columns=factor_cols)
    if F_aligned.isna().any().any():
        continue

    factor_var = float(b.T @ F_aligned @ b)
    specific_var_port = float((w.pow(2) * dvar_e).sum())
    total_var = factor_var + specific_var_port

    if not np.isfinite(total_var) or total_var <= 0:
        continue

    pred_sigma_daily = np.sqrt(total_var)
    realized_return_next = float((w * y_e).sum())

    backtest_rows.append(
        {
            "date": d,
            "next_date": next_d,
            "n_stocks": n,
            "pred_sigma_daily": pred_sigma_daily,
            "pred_sigma_annual": pred_sigma_daily * np.sqrt(TRADING_DAYS_PER_YEAR),
            "realized_return_next": realized_return_next,
            "factor_var_daily": factor_var,
            "specific_var_daily": specific_var_port,
            "factor_risk_share": factor_var / total_var,
            "specific_risk_share": specific_var_port / total_var,
            "factor_vol_annual_component": np.sqrt(max(factor_var, 0)) * np.sqrt(TRADING_DAYS_PER_YEAR),
            "specific_vol_annual_component": np.sqrt(max(specific_var_port, 0)) * np.sqrt(TRADING_DAYS_PER_YEAR),
        }
    )

backtest = pd.DataFrame(backtest_rows)

if backtest.empty:
    raise RuntimeError(
        "Backtest is empty. You may need more data before 2022 or reduce rolling windows."
    )

backtest = backtest.sort_values("date").reset_index(drop=True)
backtest["z"] = backtest["realized_return_next"] / backtest["pred_sigma_daily"]

# Realized rolling volatility of portfolio returns.
backtest["realized_vol_daily_rolling"] = (
    backtest["realized_return_next"].rolling(
        REALIZED_VOL_WINDOW, min_periods=REALIZED_VOL_WINDOW
    ).std()
)
backtest["realized_vol_annual_rolling"] = (
    backtest["realized_vol_daily_rolling"] * np.sqrt(TRADING_DAYS_PER_YEAR)
)

backtest.to_csv(OUTPUT_DIR / "backtest_results.csv", index=False)

print(f"Backtest observations: {len(backtest)}")
print(f"Backtest range: {backtest['date'].min().date()} to {backtest['date'].max().date()}")


# ============================================================
# 9. Metrics: standardized return mean/std and VaR violation rates
# ============================================================

z = backtest["z"].replace([np.inf, -np.inf], np.nan).dropna()

# One-day normal VaR on loss = -return.
loss = -backtest["realized_return_next"]
var95 = 1.6448536269514722 * backtest["pred_sigma_daily"]
var99 = 2.3263478740408408 * backtest["pred_sigma_daily"]

metrics = {
    "z_mean": float(z.mean()),
    "z_std": float(z.std(ddof=1)),
    "VaR_95_violation_rate": float((loss > var95).mean()),
    "VaR_99_violation_rate": float((loss > var99).mean()),
    "num_backtest_obs": int(len(backtest)),
    "avg_pred_sigma_daily": float(backtest["pred_sigma_daily"].mean()),
    "avg_pred_sigma_annual": float(backtest["pred_sigma_annual"].mean()),
    "avg_factor_risk_share": float(backtest["factor_risk_share"].mean()),
    "avg_specific_risk_share": float(backtest["specific_risk_share"].mean()),
}

metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv(OUTPUT_DIR / "backtest_metrics.csv", index=False)

print("\nBacktest metrics:")
for k, v in metrics.items():
    print(f"{k}: {v}")


# ============================================================
# 10. Plots
# ============================================================

# Plot 1: Predicted risk vs realized rolling volatility.
plt.figure(figsize=(12, 6))
plt.plot(backtest["date"], backtest["pred_sigma_annual"], label="Predicted annualized risk")
plt.plot(
    backtest["date"],
    backtest["realized_vol_annual_rolling"],
    label=f"Realized {REALIZED_VOL_WINDOW}-day rolling annualized vol",
)
plt.title("Predicted Risk vs Realized Rolling Volatility")
plt.xlabel("Date")
plt.ylabel("Annualized volatility")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "plot1_predicted_vs_realized_vol.png", dpi=150)
plt.close()

# Plot 2: Standardized return histogram.
plt.figure(figsize=(10, 6))
plt.hist(z, bins=40, density=True, alpha=0.8)
plt.axvline(0, linestyle="--", linewidth=1)
plt.axvline(2, linestyle="--", linewidth=1)
plt.axvline(-2, linestyle="--", linewidth=1)
plt.title("Histogram of Standardized Portfolio Returns")
plt.xlabel("z = realized return / predicted sigma")
plt.ylabel("Density")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "plot2_standardized_return_hist.png", dpi=150)
plt.close()

# Plot 3: Risk decomposition.
plt.figure(figsize=(12, 6))
plt.plot(backtest["date"], backtest["factor_risk_share"], label="Factor risk share")
plt.plot(backtest["date"], backtest["specific_risk_share"], label="Specific risk share")
plt.title("Predicted Risk Decomposition")
plt.xlabel("Date")
plt.ylabel("Share of predicted variance")
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "plot3_risk_decomposition.png", dpi=150)
plt.close()

print(f"\nAll outputs saved to: {OUTPUT_DIR}")
