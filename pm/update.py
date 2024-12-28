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

    filepath = f"{CFG.SUMMARY_DIR}/{options[i]}.csv"
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
        close = input("  Input close or enter to cancel: ")
        if close == "":
            break

        close = float(close)
        print(f"  -- Entering date: {ddate}, close: {close}")
        row = df.query("date == @ddate")
        if row.empty:
            idx = len(df)
        else:
            idx = row.index[0]
        df.loc[idx] = [ddate, close]

    print("\nSaving")
    df.to_csv(filepath, index=False)
