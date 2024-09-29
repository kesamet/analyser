import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as sco

from analyser.data import get_data


def linearfit(ts: pd.Series) -> tuple:
    """Computes linear fit of values."""
    y = ts.values
    p = np.polyfit(range(len(y)), y, deg=1)
    yfit = np.polyval(p, range(len(y)))
    residual = np.sqrt(np.mean((yfit - y) ** 2)).item()
    level = (50 + 100 * (y[-1] - yfit[-1]) / (4 * residual)).item()
    grad = (p[0] / y[0]).item()
    pred = np.polyval(p, [len(y)])[0].item()
    return yfit, level, residual, grad, pred


def compute_trend(ts: pd.Series | pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    yfit, level, residual, grad, _ = linearfit(ts)
    df = ts.to_frame() if isinstance(ts, pd.Series) else ts
    df["p0"] = yfit - residual * 2
    df["p25"] = yfit - residual
    df["p50"] = yfit
    df["p75"] = yfit + residual
    df["p100"] = yfit + residual * 2
    return df, level, grad


def plot_trend(symbol, start_date, end_date, name="", ax=None):
    """Plot time series with trends."""
    dates = pd.date_range(start_date, end_date)
    df = get_data([symbol], dates, base_symbol="ES3.SI", col="adjclose")[[symbol]]
    df, level, grad = compute_trend(df)
    close, p0, p25, p50, p75, p100 = df.iloc[-1]

    title = (
        f"{name} ({symbol}): {close:.3f} ({level:.1f}%)\n"
        f"[{p0:.3f}, {p25:.3f}, {p50:.3f}, {p75:.3f}, {p100:.3f}], {grad * 1e3:.3f}"
    )

    if ax is None:
        _, ax = plt.subplots()

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
