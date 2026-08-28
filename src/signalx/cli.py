from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from signalx import __version__
from signalx.core import DATE_COLUMN_NAMES, generate_signals
from signalx.metadata import (
    VALID_CATEGORIES,
    get_signals_by_category,
)
from signalx.utils import (
    compute_signal_stats,
    load_dataframe,
    normalize_ohlcv,
    save_dataframe,
)


def _print_stats_table(stats: dict[str, dict[str, float]]) -> None:
    """Print formatted ASCII table of signal distribution statistics."""
    if not stats:
        print("No signal columns found (*_signal).")
        return

    rows = []
    for sig_name, stat in stats.items():
        rows.append(
            {
                "Signal": sig_name,
                "Buy %": f"{stat['buy_pct']:.2f}%",
                "Sell %": f"{stat['sell_pct']:.2f}%",
                "Hold %": f"{stat['hold_pct']:.2f}%",
                "None %": f"{stat['none_pct']:.2f}%",
            }
        )
    df_stats = pd.DataFrame(rows)
    print(df_stats.to_string(index=False))


def handle_generate(args: argparse.Namespace) -> None:
    """Execute signal generation from an input OHLCV dataset."""
    input_path = Path(args.input_path)
    output_path = (
        Path(args.output)
        if args.output
        else Path("datasets") / f"{input_path.stem}_signals.parquet"
    )

    df = load_dataframe(input_path)
    print(f"Loaded {len(df)} rows from {input_path}")

    show_progress = not getattr(args, "no_progress", False)
    signals_df = generate_signals(
        df,
        drop_ohlcv=args.drop_ohlcv,
        show_progress=show_progress,
        naming=args.naming,
    )
    saved_path = save_dataframe(signals_df, output_path)
    print(
        f"Successfully generated signals. Output saved to {saved_path} "
        f"({len(signals_df)} rows, {len(signals_df.columns)} columns)"
    )

    if args.stats_report:
        print("\nSignal Distribution Summary:")
        stats = compute_signal_stats(signals_df)
        _print_stats_table(stats)


def handle_inspect(args: argparse.Namespace) -> None:
    """Inspect and validate OHLCV dataset properties and summary statistics."""
    input_path = Path(args.input_path)
    df = load_dataframe(input_path)
    normalized = normalize_ohlcv(df)

    total_rows = len(df)
    print(f"File: {input_path}")
    print(f"Total rows: {total_rows}")

    # Detect date span
    date_cols = [c for c in df.columns if str(c).strip().lower() in DATE_COLUMN_NAMES]
    if date_cols:
        date_col = date_cols[0]
        min_date = df[date_col].min()
        max_date = df[date_col].max()
        print(f"Date span: {min_date} to {max_date} (Column: '{date_col}')")
    elif isinstance(df.index, pd.DatetimeIndex):
        print(f"Date span: {df.index.min()} to {df.index.max()} (Index)")
    else:
        print("Date span: N/A (No date/datetime column found)")

    print("\nOHLCV Summary Statistics:")
    summary_data = []
    for col in ["open", "high", "low", "close", "volume"]:
        s = normalized[col]
        summary_data.append(
            {
                "Column": col,
                "Mean": f"{s.mean():.4f}",
                "Std": f"{s.std():.4f}",
                "Min": f"{s.min():.4f}",
                "Max": f"{s.max():.4f}",
            }
        )
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))


def handle_stats(args: argparse.Namespace) -> None:
    """Calculate and display distribution percentages for all *_signal columns."""
    signals_path = Path(args.signals_path)
    df = load_dataframe(signals_path)
    stats = compute_signal_stats(df)

    if args.json:
        print(json.dumps(stats, indent=2))
    else:
        print(f"Signal Statistics for {signals_path} (Total signals: {len(stats)}):")
        _print_stats_table(stats)


def handle_list(args: argparse.Namespace) -> None:
    """List signals from catalog grouped by category with triggers and descriptions."""
    if args.category:
        cat = args.category.lower().strip()
        if cat not in VALID_CATEGORIES:
            raise ValueError(
                f"Invalid category '{args.category}'. Valid categories: {sorted(list(VALID_CATEGORIES))}"
            )
        categories = [cat]
    else:
        categories = sorted(list(VALID_CATEGORIES))

    for cat in categories:
        signals = get_signals_by_category(cat)
        print(f"\n{'=' * 80}")
        print(f"Category: {cat.upper()} ({len(signals)} signals)")
        print(f"{'=' * 80}")
        for sig in signals:
            print(f"  [{sig.code}] ({sig.name})")
            print(f"    Description : {sig.description}")
            print(f"    Library     : {sig.library}")
            print(f"    Buy Trigger : {sig.buy_trigger}")
            print(f"    Sell Trigger: {sig.sell_trigger}")
            print()


_cmd_generate = handle_generate
_cmd_inspect = handle_inspect
_cmd_stats = handle_stats
_cmd_list = handle_list


def build_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser with subcommands and flags."""
    parser = argparse.ArgumentParser(
        prog="signalx",
        description="signalx: Automated Trading Signal Generation Library for Quantitative Analytics",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"signalx {__version__}",
        help="Show program version number and exit.",
    )

    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # generate subcommand
    gen_parser = subparsers.add_parser(
        "generate",
        help="Ingest OHLCV data and generate standardized trading signals.",
    )
    gen_parser.add_argument("input_path", help="Path to input OHLCV CSV or Parquet file.")
    gen_parser.add_argument(
        "-o",
        "--output",
        help="Path to output CSV or Parquet file (default: datasets/<stem>_signals.parquet).",
    )
    gen_parser.add_argument(
        "--drop-ohlcv",
        action="store_true",
        help="Drop OHLCV columns and retain only date and signal columns.",
    )
    gen_parser.add_argument(
        "--naming",
        type=str,
        choices=["code", "semantic"],
        default="code",
        help="Signal naming convention: code (e.g. TRD001_signal) or semantic (e.g. trend_sma_cross_5_20_signal). Default: code.",
    )
    gen_parser.add_argument(
        "--stats-report",
        action="store_true",
        help="Print signal distribution statistics after generation.",
    )
    gen_parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable real-time progress bars during signal generation.",
    )

    # inspect subcommand
    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Validate OHLCV columns and display dataset summary statistics.",
    )
    inspect_parser.add_argument("input_path", help="Path to input OHLCV CSV or Parquet file.")

    # stats subcommand
    stats_parser = subparsers.add_parser(
        "stats",
        help="Calculate signal distribution percentages (buy/sell/hold/none).",
    )
    stats_parser.add_argument(
        "signals_path",
        help="Path to generated signals CSV or Parquet file.",
    )
    stats_parser.add_argument(
        "--json",
        action="store_true",
        help="Output distribution statistics formatted as JSON.",
    )

    # list subcommand
    list_parser = subparsers.add_parser(
        "list",
        help="List available trading signals and metadata from SIGNAL_CATALOG.",
    )
    list_parser.add_argument(
        "-c",
        "--category",
        help="Filter signals by category (e.g. trend, momentum, volatility, volume, candlestick, statistical, composite).",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for signalx command-line application."""
    parser = build_parser()
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        parser.print_help()
        return

    args = parser.parse_args(argv)

    try:
        if args.subcommand == "generate":
            handle_generate(args)
        elif args.subcommand == "inspect":
            handle_inspect(args)
        elif args.subcommand == "stats":
            handle_stats(args)
        elif args.subcommand == "list":
            handle_list(args)
        else:
            parser.print_help()
    except (FileNotFoundError, ValueError) as err:
        sys.stderr.write(f"Error: {err}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
