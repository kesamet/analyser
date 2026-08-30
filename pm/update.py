"""
Script to update csv.
"""

from datetime import timedelta

import pandas as pd
import questionary
from dateutil import parser

from pm import CFG


def select_fund() -> str:
    """Select a fund from the available options."""
    choices = [
        questionary.Choice(title=f"{i}: {fund}", value=fund) for i, fund in enumerate(CFG.FUNDS)
    ]
    fund = questionary.select(
        "Select fund to update:",
        choices=choices,
    ).ask()
    if fund is None:
        raise KeyboardInterrupt("Selection cancelled")
    return fund


def get_periods() -> int:
    """Get number of periods to amend."""
    periods_str = questionary.text(
        "Enter number of periods to amend (default=1):",
        default="1",
        validate=lambda x: x.isdigit() or "Please enter a valid number",
    ).ask()
    if periods_str is None:
        raise KeyboardInterrupt("Input cancelled")
    return int(periods_str) if periods_str else 1


def get_close_price(date_str: str, is_usd_fund: bool = False) -> float | None:
    """Get close price input from user."""
    prompt = f"Date: {date_str}\nInput close price (or press Enter to cancel): "
    if is_usd_fund:
        prompt = f"Date: {date_str}\nInput USD close price (or press Enter to cancel): "

    close_str = questionary.text(
        prompt,
        validate=lambda x: (
            x == "" or x.replace(".", "", 1).isdigit() or "Please enter a valid number"
        ),
    ).ask()
    if close_str is None or close_str == "":
        return None
    return float(close_str)


def main():
    sheet = select_fund()

    if sheet in CFG.USE_USD:
        usd_sgd = pd.read_csv(f"{CFG.DATA_DIR}/USDSGD=X.csv")

    filepath = f"{CFG.SUMMARY_DIR}/{sheet}.csv"
    df = pd.read_csv(filepath)
    print(df.tail())

    periods = get_periods()

    if sheet in CFG.MONTHLY:
        _date = parser.parse(df["date"].iloc[-1]).replace(day=1) - pd.DateOffset(months=periods)
    else:
        _date = parser.parse(df["date"].iloc[-1]) - timedelta(days=periods)

    while True:
        # Monthly updates for certain funds: first of month
        if sheet in CFG.MONTHLY:
            _date = (_date + pd.DateOffset(months=1)).replace(day=1)
        else:
            _date += timedelta(days=1)

        # Skip weekends
        while True:
            dow = _date.strftime("%A")
            if dow in ["Saturday", "Sunday"]:
                _date += timedelta(days=1)
            else:
                break
        date_str = _date.strftime("%Y-%m-%d")

        # For USD based funds, get USDSGD close
        usd_sgd_close = None
        if sheet in CFG.USE_USD:
            curr_date = _date
            curr_date_str = curr_date.strftime("%Y-%m-%d")
            while True:
                row = usd_sgd.query("date == @curr_date_str")
                if row.empty:
                    curr_date -= timedelta(days=1)
                    curr_date_str = curr_date.strftime("%Y-%m-%d")
                else:
                    print(f"  -- Using close on {curr_date_str}")
                    usd_sgd_close = row["close"].values[0]
                    print(f"  -- USDSGD close: {usd_sgd_close:.4f}")
                    break

        row = df.query("date == @date_str")
        if row.empty:
            idx = len(df)
        else:
            idx = row.index[0]

        close = get_close_price(date_str, sheet in CFG.USE_USD)
        if close is None:
            break

        if sheet in CFG.USE_USD:
            usd_close = close
            close = round(usd_close * usd_sgd_close, 2)
            print(f"  -- Entering date: {date_str}, close: {close}, usd_close: {usd_close}")
            df.loc[idx] = [date_str, close, usd_close]
        else:
            print(f"  -- Entering date: {date_str}, close: {close}")
            df.loc[idx] = [date_str, close]

    print("\nSaving")
    df.to_csv(filepath, index=False)


if __name__ == "__main__":
    main()
