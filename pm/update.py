"""
Script to update csv.
"""

from datetime import timedelta
from dateutil import parser

import pandas as pd

from pm import CFG


if __name__ == "__main__":
    options = ["SRS", "Core", "CoreEnhanced", "QGF", "HGPS"]
    i = int(input("  Enter sheet (SRS=0, Core=1, CoreEnhanced=2, QGF=3, HGPS=4): "))
    if i not in range(len(options)):
        raise IndexError

    _data = options[i]
    if _data == "CoreEnhanced":
        usd_sgd = pd.read_csv(f"./data/USDSGD=X.csv")

    filepath = f"{CFG.SUMMARY_DIR}/{_data}.csv"
    df = pd.read_csv(filepath)
    print(df.tail())

    days = input("\n  Enter ndays to amend (default=1): ")
    days = 1 if days == "" else int(days)
    _date = parser.parse(df["date"].iloc[-1]) - timedelta(days=days)
    while True:
        while True:
            _date += timedelta(days=1)
            ddate = _date.strftime("%Y-%m-%d")
            dow = _date.strftime("%A")
            if dow not in ["Saturday", "Sunday"]:
                break

        print(f"\nDate: {ddate} {dow}")
        if _data == "CoreEnhanced":
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
        if _data == "CoreEnhanced":
            usd_close = close
            close = round(usd_close * usd_sgd_close, 2)
            print(f"  -- Entering date: {ddate}, close: {close}, usd_close: {usd_close}")
            df.loc[idx] = [ddate, close, usd_close]
        else:
            print(f"  -- Entering date: {ddate}, close: {close}")
            df.loc[idx] = [ddate, close]

    print("\nSaving")
    df.to_csv(filepath, index=False)
