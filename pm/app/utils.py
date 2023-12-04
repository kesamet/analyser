from datetime import datetime, timedelta
from typing import Tuple

import streamlit as st


def get_start_date(
    today_date: datetime,
    start_date: str = "2015-01-01",
    options: Tuple[str] = ("YTD", "1M", "6M", "1Y", "2Y", "3Y", "All time"),
) -> str:
    select_range = st.selectbox("Select time range", options)
    if select_range[-1] == "Y":
        _yr = today_date.year - int(select_range[:-1])
        return today_date.replace(year=_yr).strftime("%Y-%m-%d")
    elif select_range[-1] == "M":
        _mths = int(select_range[:-1])
        return (today_date - timedelta(days=30 * _mths)).strftime("%Y-%m-%d")
    elif select_range == "YTD":
        return today_date.replace(month=1, day=1).strftime("%Y-%m-%d")
    elif select_range == "MTD":
        return today_date.replace(day=1).strftime("%Y-%m-%d")
    return start_date
