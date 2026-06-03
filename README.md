# Barra Equity Risk Model

This project implements a Barra-style equity risk model for a selected universe of U.S. stocks. The model uses industry factors and style factors to estimate factor returns, forecast portfolio risk, decompose predicted risk, and evaluate the model through backtesting.

## Project Overview

The goal of this project is to build an equity risk model similar in spirit to the Barra framework. The model reads daily stock price and volume data, constructs factor exposures, estimates factor returns through daily cross-sectional regressions, builds a covariance-based risk forecast, and evaluates the predicted risk using realized portfolio returns.

The project includes:

* Stock universe preparation
* Industry factor construction using GICS sectors
* Style factor construction
* Daily cross-sectional factor return estimation
* Factor covariance matrix estimation
* Specific risk estimation
* Ex-ante portfolio risk prediction
* VaR backtesting
* Risk decomposition visualization

## Repository Structure

```text
barra-equity-risk-model/
│
├── select_stock_file.py
├── barra_model_project.py
├── README.md
│
├── target stocks/
│   └── selected stock .txt files
│
├── barra_outputs/
│   ├── factor_returns.csv
│   ├── specific_residuals.csv
│   ├── backtest_results.csv
│   ├── backtest_metrics.csv
│   ├── missing_files.txt
│   ├── plot1_predicted_vs_realized_vol.png
│   ├── plot2_standardized_return_hist.png
│   └── plot3_risk_decomposition.png
```

## Data Preparation

The script `select_stock_file.py` is used to prepare the target stock universe. It reads a list of expected stock filenames from:

```text
barra_sp500_500_expected_filenames.txt
```

Then it searches for these files in the following source folders:

```text
nasdaq stocks/
nyse stocks/
```

If a target stock file is found, it is copied into:

```text
target stocks/
```

If a file is missing, its filename is saved into:

```text
target stocks/missing_files.txt
```

This step creates a clean folder containing only the stock files needed for the Barra risk model.

## Risk Model Inputs

The main script `barra_model_project.py` expects the following inputs:

```text
target stocks/
barra_sp500_500_expected_filenames.txt
barra_sp500_500_gics_sectors.txt
```

Each stock file is expected to contain daily data with columns similar to:

```text
<TICKER>, <PER>, <DATE>, <TIME>, <OPEN>, <HIGH>, <LOW>, <CLOSE>, <VOL>, <OPENINT>
```

The model uses daily close prices and volume data from 2022 to 2025.

## Factor Construction

### Industry Factors

Industry factors are constructed using GICS sector information. Each stock is assigned to one GICS sector, and the model converts these sector labels into dummy variables.

For example, if a stock belongs to the Information Technology sector, its exposure to the Information Technology industry factor is 1, while its exposure to other industry factors is 0.

### Style Factors

The model constructs four style factors:

1. **Momentum**

   Momentum is calculated using past 252 trading days of log returns, while skipping the most recent 21 trading days.

2. **Volatility**

   Volatility is calculated as the 252-day rolling standard deviation of daily log returns.

3. **Liquidity**

   Liquidity is measured using the log of the 21-day average dollar volume.

4. **Beta**

   Beta is calculated as the 252-day rolling beta of each stock against an equal-weighted market return.

All style factors are standardized cross-sectionally by date, so that each factor has approximately zero mean and unit standard deviation across the stock universe.

## Factor Return Estimation

The model estimates daily factor returns using cross-sectional regression:

```text
r_{i,t+1} = X_{i,t} f_{t+1} + epsilon_{i,t+1}
```

where:

* `r_{i,t+1}` is the next-day return of stock `i`
* `X_{i,t}` is the factor exposure vector of stock `i` at date `t`
* `f_{t+1}` is the estimated factor return vector
* `epsilon_{i,t+1}` is the stock-specific residual return

No intercept is used because all GICS sector dummy variables are included.

## Risk Forecasting

The model forecasts the next-day portfolio variance using the standard factor risk model structure:

```text
Sigma_t = X_t F_t X_t' + D_t
```

where:

* `X_t` is the stock-by-factor exposure matrix
* `F_t` is the factor covariance matrix
* `D_t` is the diagonal specific risk matrix

The model then predicts the risk of an equal-weighted portfolio over all eligible stocks.

The predicted portfolio variance is decomposed into:

```text
Total variance = Factor variance + Specific variance
```

## Backtesting

The model backtests the predicted risk using realized next-day portfolio returns.

The standardized return is calculated as:

```text
z_t = realized return / predicted daily sigma
```

If the model is well calibrated, the standardized returns should have:

```text
mean close to 0
standard deviation close to 1
```

The model also calculates one-day normal VaR:

```text
VaR_95 = 1.645 × predicted daily sigma
VaR_99 = 2.326 × predicted daily sigma
```

Then it evaluates the VaR violation rates:

```text
VaR 95% violation rate = frequency of loss > VaR_95
VaR 99% violation rate = frequency of loss > VaR_99
```

For a well-calibrated model:

```text
VaR 95% violation rate should be close to 5%
VaR 99% violation rate should be close to 1%
```

## Outputs

The script generates the following output files:

```text
barra_outputs/factor_returns.csv
barra_outputs/specific_residuals.csv
barra_outputs/backtest_results.csv
barra_outputs/backtest_metrics.csv
barra_outputs/missing_files.txt
```

It also generates three plots:

```text
plot1_predicted_vs_realized_vol.png
plot2_standardized_return_hist.png
plot3_risk_decomposition.png
```

### Plot 1: Predicted Risk vs Realized Rolling Volatility

This plot compares the model-predicted annualized volatility with realized rolling annualized volatility.

### Plot 2: Standardized Return Histogram

This plot shows the distribution of standardized portfolio returns:

```text
z = realized return / predicted sigma
```

It is used to check whether the predicted risk is well calibrated.

### Plot 3: Risk Decomposition

This plot shows the predicted factor risk share and specific risk share over time.

## How to Run

First, update the base directory path in both Python scripts:

```python
BASE_DIR = Path(r"C:\Users\caoxi\python练习\Barra")
```

or:

```python
base_dir = Path(r"C:\Users\caoxi\python练习\Barra")
```

Then run the stock selection script:

```bash
python select_stock_file.py
```

After the target stock files are prepared, run the main Barra model script:

```bash
python barra_model_project.py
```

All outputs will be saved to:

```text
barra_outputs/
```

## Main Python Libraries

This project uses:

* `pandas`
* `numpy`
* `matplotlib`
* `pathlib`
* `shutil`

## Project Purpose

This project is designed as a quantitative risk management project. It demonstrates how to construct a factor-based equity risk model, estimate factor returns, forecast portfolio volatility, decompose risk sources, and validate risk forecasts through standardized returns and VaR backtesting.
