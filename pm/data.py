import pandas as pd

from pm import CFG


def load_ie_data(start_date: str = "1990-01-01") -> pd.DataFrame:
    """Data downloaded from https://shillerdata.com/."""
    df = pd.read_excel(f"{CFG.SUMMARY_DIR}/ie_data.xls", sheet_name="Data", skiprows=7)
    df.drop(["Fraction", "Unnamed: 13", "Unnamed: 15"], axis=1, inplace=True)
    df.columns = [
        "Date",
        "S&P500",
        "Dividend",
        "Earnings",
        "CPI",
        "Long_IR",
        "Real_Price",
        "Real_Dividend",
        "Real_TR_Price",
        "Real_Earnings",
        "Real_TR_Scaled_Earnings",
        "CAPE",
        "TRCAPE",
        "Excess_CAPE_Yield",
        "Mth_Bond_TR",
        "Bond_RTR",
        "10Y_Stock_RR",
        "10Y_Bond_RR",
        "10Y_Excess_RR",
    ]

    df["Date"] = df["Date"].astype(str)
    df["Date"] = df["Date"].apply(lambda x: x + "0" if len(x) == 6 else x)
    df["Date"] = pd.to_datetime(df["Date"].astype(str), format="%Y.%m")
    df.set_index("Date", inplace=True)

    df = df.iloc[:-1]
    if start_date is not None:
        df = df[df.index >= start_date]

    df["10xReal_Earnings"] = 10 * df["Real_Earnings"]
    df["10xLong_IR"] = 10 * df["Long_IR"]
    return df[["Real_Price", "10xReal_Earnings", "CAPE", "10xLong_IR"]]


def load_pe_data(start_date: str = "1990-01-01") -> pd.DataFrame:
    """Shiller monthly PE data downloaded from nasdaq-data-link."""
    df = pd.read_csv(f"{CFG.SUMMARY_DIR}/pe_data.csv")
    df["Date"] = pd.to_datetime(df["Date"].astype(str))
    df.set_index("Date", inplace=True)
    if start_date is not None:
        df = df[df.index >= start_date]
    return df
