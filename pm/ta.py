import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as sco

from analyser.data import get_data


def linearfit(ts):
    """Computes linear fit of values."""
    y = ts.values
    p = np.polyfit(range(len(y)), y, deg=1)
    yfit = np.polyval(p, range(len(y)))
    last = yfit[-1]
    residual = np.sqrt(np.mean((yfit - y) ** 2))
    level = 50 + 100 * (y[-1] - last) / (4 * residual)
    grad = p[0] / y[0]
    pred = np.polyval(p, [len(y)])[0]
    return yfit, level, residual, last, grad, pred


def compute_trend(dates, symbol):
    """Compute linear trend."""
    if symbol in ["EIMI.L", "IWDA.L"]:
        df = get_data([symbol], dates, base_symbol="USDSGD=X", col="close")[[symbol]]
    else:
        df = get_data([symbol], dates, base_symbol="ES3.SI", col="adjclose")[[symbol]]

    yfit, level, residual, last, grad, pred = linearfit(df[symbol])
    df["p0"] = yfit - residual * 2
    df["p25"] = yfit - residual
    df["p50"] = yfit
    df["p75"] = yfit + residual
    df["p100"] = yfit + residual * 2
    # pred_row = [pred - 2 * residual, pred - residual, pred, pred + residual, pred + 2 * residual]
    return df, level, grad


def plot_trend(symbol, start_date, end_date, name="", ax=None):
    """Plot time series with trends."""
    dates = pd.date_range(start_date, end_date)
    df, level, grad = compute_trend(dates, symbol)
    close, p0, p25, p50, p75, p100 = df.iloc[-1]

    title = f"""
        {name} ({symbol}): {close:.3f} ({level:.1f}%)
        [{p0:.3f}, {p25:.3f}, {p50:.3f}, {p75:.3f}, {p100:.3f}], {grad * 1e3:.3f}
        """

    if ax is None:
        fig, ax = plt.subplots()

    df[symbol].plot(color="blue", ax=ax)
    df["p0"].plot(color="green", ax=ax)
    df["p25"].plot(color="green", ax=ax)
    df["p50"].plot(color="green", ax=ax)
    df["p75"].plot(color="red", ax=ax)
    df["p100"].plot(color="red", ax=ax)
    ax.set_title(title)
    return ax


def compute_dietz_ret(df: pd.DataFrame) -> np.float32:
    """Compute modified Dietz return."""
    cf = (
        df["Cost"].diff().dropna().to_numpy()
        - df["Realised_Gain"].iloc[1:].to_numpy()
        - df["Div"].iloc[1:].to_numpy()
    )
    t = np.linspace(1, 0, len(cf) + 1)[1:]
    r = (df["Portfolio"].iloc[-1] - df["Portfolio"].iloc[0] - cf.sum()) / (
        df["Portfolio"].iloc[0] + t.dot(cf)
    )
    return r


def compute_xnpv(cashflows: np.ndarray, rate: float) -> np.float32:
    """Compute the net present value of a series of cashflows
    at irregular intervals.

    Args:
        cashflows: pandas.Series of values with dates as index
        rate: risk-free rate

    Returns:
        NPV of the given cash flows
    """
    arr = cashflows.reset_index().values
    t0 = arr[0, 0]
    return np.sum([r[1] / (1 + rate) ** ((r[0] - t0).days / 365.25) for r in arr])


def compute_xirr(cashflows: np.ndarray, initial: float = 0.1) -> np.float32:
    """Compute internal rate of return of a series of cashflows
    at irregular intervals.

    Args:
        cashflows: numpy.array of datetimes and values
        initial: initial guess

    Returns:
        XIRR
    """
    return sco.newton(lambda r: compute_xnpv(cashflows, r), initial)
