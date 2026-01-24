"""
Script to update csv.
"""

from datetime import timedelta
from dateutil import parser

import pandas as pd

from pm import CFG

USE_USD = ["CoreEnhanced", "QGF", "HGPS", "EPC"]


def main():
    options = ["SRS", "Core", "CoreEnhanced", "QGF", "HGPS", "EPC", "Gold"]
    i = int(input("  Enter sheet (SRS=0, Core=1, CoreEnhanced=2, QGF=3, HGPS=4, EPC=5, Gold=6): "))
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
        print(f"\nDate: {date_str}")

        curr_date = _date
        curr_date_str = curr_date.strftime("%Y-%m-%d")
        if sheet in USE_USD:
            while True:
                row = usd_sgd.query("date == @curr_date_str")
                if not row.empty:
                    print(f"  -- Using close on {curr_date_str}")
                    usd_sgd_close = row["close"].values[0]
                    break
                else:
                    curr_date -= timedelta(days=1)
                    curr_date_str = curr_date.strftime("%Y-%m-%d")
            print(f"  -- USDSGD close: {usd_sgd_close:.4f}")

        row = df.query("date == @date_str")
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
            print(f"  -- Entering date: {date_str}, close: {close}, usd_close: {usd_close}")
            df.loc[idx] = [date_str, close, usd_close]
        else:
            print(f"  -- Entering date: {date_str}, close: {close}")
            df.loc[idx] = [date_str, close]

    print("\nSaving")
    df.to_csv(filepath, index=False)


if __name__ == "__main__":
    main()
