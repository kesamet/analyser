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

    _date = parser.parse(df["date"].iloc[-1])
    _date -= timedelta(days=1)
    while True:
        while True:
            _date += timedelta(days=1)
            date = _date.strftime("%Y-%m-%d")
            dow = _date.strftime("%A")
            if dow not in ["Saturday", "Sunday"]:
                break

        print(f"\nDate: {date} {dow}")
        close = input("  Input close or enter to cancel: ")
        if close == "":
            break

        close = float(close)
        print(f"  -- Entering date: {date}, close: {close}")
        row = df.query("date == @date")
        if row.empty:
            idx = len(df)
        else:
            idx = row.index[0]
        df.loc[idx] = [date, close]

    print("\nSaving")
    df.to_csv(filepath, index=False)
