#!/usr/bin/env python3
"""
SignalX: Prepare Sample OHLCV Datasets
Generates realistic synthetic OHLCV time-series data for testing and demonstration.
Outputs:
  - datasets/sample_ohlcv.parquet
  - datasets/sample_ohlcv.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_realistic_ohlcv(
    n_bars: int = 1000,
    initial_price: float = 100.0,
    volatility: float = 0.02,
    drift: float = 0.0003,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic synthetic OHLCV data using geometric Brownian motion with noise."""
    rng = np.random.default_rng(seed)

    # Generate log returns
    dt = 1.0
    random_shocks = rng.normal(
        loc=(drift - 0.5 * volatility**2) * dt,
        scale=volatility * np.sqrt(dt),
        size=n_bars,
    )

    log_prices = np.log(initial_price) + np.cumsum(random_shocks)
    close_prices = np.exp(log_prices)

    # Generate open, high, low around close prices
    intraday_volatility = rng.uniform(0.005, 0.025, size=n_bars)

    open_prices = np.zeros(n_bars)
    open_prices[0] = initial_price
    for i in range(1, n_bars):
        # Open is previous close with small overnight gap
        gap = rng.normal(0, 0.003) * close_prices[i - 1]
        open_prices[i] = max(0.01, close_prices[i - 1] + gap)

    high_prices = np.maximum(open_prices, close_prices) + np.abs(
        rng.normal(0, intraday_volatility * close_prices)
    )
    low_prices = np.minimum(open_prices, close_prices) - np.abs(
        rng.normal(0, intraday_volatility * close_prices)
    )
    # Ensure Low > 0
    low_prices = np.maximum(low_prices, 0.01)
    # Ensure High >= max(Open, Close) and Low <= min(Open, Close)
    high_prices = np.maximum(high_prices, np.maximum(open_prices, close_prices))
    low_prices = np.minimum(low_prices, np.minimum(open_prices, close_prices))

    # Generate realistic volume with lognormal distribution and occasional spikes
    base_volume = rng.lognormal(mean=12.0, sigma=0.6, size=n_bars)
    spike_mask = rng.random(n_bars) < 0.05
    base_volume[spike_mask] *= rng.uniform(2.5, 6.0, size=np.sum(spike_mask))
    volume = np.round(base_volume).astype(np.int64)

    # Generate Date index
    start_date = pd.Timestamp("2023-01-01 09:30:00")
    dates = pd.date_range(start=start_date, periods=n_bars, freq="1D")

    df = pd.DataFrame(
        {
            "Date": dates,
            "Open": np.round(open_prices, 2),
            "High": np.round(high_prices, 2),
            "Low": np.round(low_prices, 2),
            "Close": np.round(close_prices, 2),
            "Volume": volume,
        }
    )
    return df


def main() -> None:
    dataset_dir = Path("datasets")
    dataset_dir.mkdir(parents=True, exist_ok=True)

    print("Generating 1,000 bars of realistic OHLCV data...")
    df = generate_realistic_ohlcv(n_bars=1000, seed=42)

    parquet_path = dataset_dir / "sample_ohlcv.parquet"
    csv_path = dataset_dir / "sample_ohlcv.csv"

    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)

    print(f"Saved: {parquet_path} ({len(df)} rows, {parquet_path.stat().st_size:,} bytes)")
    print(f"Saved: {csv_path} ({len(df)} rows, {csv_path.stat().st_size:,} bytes)")
    print("\nFirst 5 rows:")
    print(df.head())


if __name__ == "__main__":
    main()
