"""
Script to update csv.
"""

from datetime import timedelta
from dateutil import parser

import pandas as pd

from pm import CFG


def main():
    x = ", ".join([f"{f}={i}" for i, f in enumerate(CFG.FUNDS)])
    i = int(input(f"  Enter sheet ({x}): "))
    if i not in range(len(CFG.FUNDS)):
        raise IndexError

    sheet = CFG.FUNDS[i]
    if sheet in CFG.USE_USD:
        usd_sgd = pd.read_csv(f"{CFG.DATA_DIR}/USDSGD=X.csv")

    filepath = f"{CFG.SUMMARY_DIR}/{sheet}.csv"
    df = pd.read_csv(filepath)
    print(df.tail())

    periods = input("\n  Enter num periods to amend (default=1): ")
    periods = 1 if periods == "" else int(periods)
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
        print(f"\nDate: {date_str}")

        # For USD based funds, get USDSGD close
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
                    break
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

    # For Gold fund, aggregate SGD and USD
    if sheet.startswith("Gold"):
        df1 = pd.read_csv(f"{CFG.SUMMARY_DIR}/Gold_SGD.csv")
        df2 = pd.read_csv(f"{CFG.SUMMARY_DIR}/Gold_USD.csv")
        df_merged = pd.merge(df1, df2, on="date", suffixes=("_SGD", "_USD"), how="outer").fillna(0)
        df_merged["close"] = df_merged["close_SGD"] + df_merged["close_USD"]
        df_merged = df_merged[["date", "close"]]
        print("Gold aggregated:")
        print(df_merged.tail())
        df_merged.to_csv(f"{CFG.SUMMARY_DIR}/Gold.csv", index=False)


if __name__ == "__main__":
    main()
