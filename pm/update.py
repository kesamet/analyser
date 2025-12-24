"""
Script to update csv.
"""

from datetime import timedelta
from dateutil import parser

import pandas as pd

from pm import CFG

USE_USD = ["CoreEnhanced", "QGF", "HGPS", "EPC"]


def main():
    options = ["SRS", "Core", "CoreEnhanced", "QGF", "HGPS", "EPC"]
    i = int(input("  Enter sheet (SRS=0, Core=1, CoreEnhanced=2, QGF=3, HGPS=4, EPC=5): "))
    if i not in range(len(options)):
        raise IndexError

    sheet = options[i]
    if sheet in USE_USD:
        usd_sgd = pd.read_csv(f"{CFG.DATA_DIR}/USDSGD=X.csv")

    filepath = f"{CFG.SUMMARY_DIR}/{sheet}.csv"
    df = pd.read_csv(filepath)
    print(df.tail())

    periods = input("\n  Enter num periods to amend (default=1): ")
    periods = 1 if periods == "" else int(periods)
    if sheet in ["QGF", "HGPS", "EPC"]:
        _date = parser.parse(df["date"].iloc[-1]).replace(day=1) - pd.DateOffset(months=periods)
    else:
        _date = parser.parse(df["date"].iloc[-1]) - timedelta(days=periods)

    while True:
        if sheet in ["QGF", "HGPS", "EPC"]:
            _date = _date.replace(day=1)
            _date += pd.DateOffset(months=periods)
        else:
            _date += timedelta(days=periods)

        # Skip weekends
        while True:
            dow = _date.strftime("%A")
            if dow in ["Saturday", "Sunday"]:
                _date += timedelta(days=1)
            else:
                break

        ddate = _date.strftime("%Y-%m-%d")
        print(f"\nDate: {ddate} {dow}")
        if sheet in USE_USD:
            row = usd_sgd.query("date == @ddate")
            if row.empty:
                print("  -- USDSGD data not available, skipping")
                break
            usd_sgd_close = row["close"].values[0]
            print(f"  -- USDSGD close: {usd_sgd_close:.4f}")

        row = df.query("date == @ddate")
        if row.empty:
            idx = len(df)
        else:
            idx = row.index[0]

        close = input("  Input close or enter to cancel: ")
        if close == "":
            break

        close = float(close)
        if sheet in USE_USD:
            usd_close = close
            close = round(usd_close * usd_sgd_close, 2)
            print(f"  -- Entering date: {ddate}, close: {close}, usd_close: {usd_close}")
            df.loc[idx] = [ddate, close, usd_close]
        else:
            print(f"  -- Entering date: {ddate}, close: {close}")
            df.loc[idx] = [ddate, close]

    print("\nSaving")
    df.to_csv(filepath, index=False)


if __name__ == "__main__":
    main()
