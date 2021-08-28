"""
Script to compute efficient frontier using Modern Portfolio Theory.
"""
import numpy as np
import pandas as pd
import scipy.optimize as sco
import matplotlib.pyplot as plt
from IPython.display import display

plt.style.use("seaborn-darkgrid")
# plt.style.use("fivethirtyeight")


def portfolio_vol(weights, cov_mat):
    """Compute portfolio annualised volatility."""
    return np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights))) * np.sqrt(252)


def portfolio_ret(weights, avg_rets):
    """Compute portfolio annualised average return."""
    return np.dot(avg_rets, weights) * 252


def sharpe(weights, avg_rets, cov_mat, rfr):
    """Compute Sharpe ratio."""
    return (portfolio_ret(weights, avg_rets) - rfr) / portfolio_vol(weights, cov_mat)


def m2(weights, avg_rets, cov_mat, rfr, bm_vol):
    """Compute Modigliani risk-adjusted performance."""
    return sharpe(weights, avg_rets, cov_mat, rfr) * bm_vol + rfr


def portfolio_perf(weights, avg_rets, cov_mat, rfr, bm_vol):
    """Compute porfolio performance
    (return, volatility, Sharpe ratio, Modigliani risk-adjusted performance)."""
    ret = portfolio_ret(weights, avg_rets)
    vol = portfolio_vol(weights, cov_mat)
    sha = (ret - rfr) / vol
    return ret, vol, sha, sha * bm_vol + rfr


def max_sharpe(avg_rets, cov_mat, rfr):
    """Maximize Sharpe ratio."""
    def neg_sharpe(weights, avg_rets, cov_mat, rfr):
        return -sharpe(weights, avg_rets, cov_mat, rfr)

    num_assets = len(avg_rets)
    return sco.minimize(
        neg_sharpe,
        np.ones(num_assets) / num_assets,
        args=(avg_rets, cov_mat, rfr),
        method="SLSQP",
        bounds=((0.0, 1.0),) * num_assets,
        constraints=({"type": "eq", "fun": lambda x: np.sum(x) - 1}))


def min_vol(cov_mat):
    """Minimize volatility."""
    num_assets = len(cov_mat.columns)
    return sco.minimize(
        portfolio_vol,
        np.ones(num_assets) / num_assets,
        args=(cov_mat,),
        method="SLSQP",
        bounds=((0.0, 1.0),) * num_assets,
        constraints=({"type": "eq", "fun": lambda x: np.sum(x) - 1}))


def eff_vol(avg_rets, cov_mat, target):
    """Compute efficient volatility given target return."""
    def pret(weights):
        return portfolio_ret(weights, avg_rets)

    num_assets = len(cov_mat.columns)
    return sco.minimize(
        portfolio_vol,
        np.ones(num_assets) / num_assets,
        args=(cov_mat,),
        method="SLSQP",
        bounds=((0.0, 1.0),) * num_assets,
        constraints=({"type": "eq", "fun": lambda x: pret(x) - target},
                     {"type": "eq", "fun": lambda x: np.sum(x) - 1}))


def eff_ret(avg_rets, cov_mat, target):
    """Compute efficient return given target volatility."""
    def pvol(weights):
        return portfolio_vol(weights, cov_mat)

    def neg_pret(weights, avg_rets):
        return -portfolio_ret(weights, avg_rets)

    num_assets = len(cov_mat.columns)
    return sco.minimize(
        neg_pret,
        np.ones(num_assets) / num_assets,
        args=(avg_rets,),
        method="SLSQP",
        bounds=((0.0, 1.0),) * num_assets,
        constraints=({"type": "eq", "fun": lambda x: pvol(x) - target},
                     {"type": "eq", "fun": lambda x: np.sum(x) - 1}))


def efrontier_vol(avg_rets, cov_mat, rets_range):
    """Sample efficient frontier given returns."""
    efficients = []
    for ret in rets_range:
        ef = eff_vol(avg_rets, cov_mat, ret)
        if ef["success"]:
            efficients.append([ef["fun"], ret])
        else:
            break
    efficients = pd.DataFrame(efficients, columns=["vol", "ret"])
    return efficients


def efrontier_ret(avg_rets, cov_mat, vols_range):
    """Sample efficient frontier given volatilities."""
    efficients = []
    for vol in vols_range:
        ef = eff_ret(avg_rets, cov_mat, vol)
        if ef["success"]:
            efficients.append([vol, -ef["fun"]])
        else:
            break
    efficients = pd.DataFrame(efficients, columns=["vol", "ret"])
    return efficients


def allocations(allocs, names):
    df = pd.DataFrame(np.round(100*allocs, 2), index=names, columns=["alloc%"])
    return df.T


def display_ef(avg_rets, cov_mat, rfr, bm_vol):
    """Display efficient frontier."""
    colnames = avg_rets.index

    max_sharpe_val = max_sharpe(avg_rets, cov_mat, rfr)["x"]
    max_sharpe_ret, max_sharpe_vol, max_sharpe_sharpe, max_sharpe_m2 = portfolio_perf(
        max_sharpe_val, avg_rets, cov_mat, rfr, bm_vol)

    min_vol_val = min_vol(cov_mat)["x"]
    min_vol_ret, min_vol_vol, min_vol_sharpe, min_vol_m2 = portfolio_perf(
        min_vol_val, avg_rets, cov_mat, rfr, bm_vol)

    ann_vol = np.sqrt(np.diag(cov_mat)) * np.sqrt(252)
    ann_ret = avg_rets * 252

    print("-"*80)
    print("Maximum Sharpe Ratio Portfolio Allocation\n")
    print("M2: {:.2f}".format(max_sharpe_m2))
    print("Sharpe ratio: {:.2f}".format(max_sharpe_sharpe))
    print("Annualised Return: {:.2f}".format(max_sharpe_ret))
    print("Annualised Volatility: {:.2f}\n".format(max_sharpe_vol))
    display(allocations(max_sharpe_val, colnames))

    print("-"*80)
    print("Minimum Volatility Portfolio Allocation\n")
    print("M2: {:.2f}".format(min_vol_m2))
    print("Sharpe ratio: {:.2f}".format(min_vol_sharpe))
    print("Annualised Return: {:.2f}".format(min_vol_ret))
    print("Annualised Volatility: {:.2f}\n".format(min_vol_vol))
    display(allocations(min_vol_val, colnames))

    print("-"*80)
    print("Individual Stock Returns and Volatility\n")
    for i, txt in enumerate(colnames):
        print("{}: return {:.2f}, volatility: {:.2f}"
              .format(txt, ann_ret[i], ann_vol[i]))

    print("-"*80)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(ann_vol, ann_ret, marker="o", s=100)

    for i, txt in enumerate(colnames):
        ax.annotate(txt, (ann_vol[i], ann_ret[i]), xytext=(10, 0),
                    textcoords="offset points")
    ax.scatter(max_sharpe_vol, max_sharpe_ret, marker="*", color="r", s=200,
               label="Max Sharpe ratio")
    ax.scatter(min_vol_vol, min_vol_ret, marker="*", color="g", s=200,
               label="Min volatility")

    targets = np.linspace(min_vol_ret, min_vol_ret*6, 20)
    eff_ports = efrontier_vol(avg_rets, cov_mat, targets)
    ax.plot(eff_ports["vol"], eff_ports["ret"], linestyle="-.", color="k",
            label="efficient frontier")

    ax.set_title("Portfolio Optimization with Individual Stocks")
    ax.set_xlabel("annualised volatility")
    ax.set_ylabel("annualised returns")
    ax.legend(labelspacing=0.8)
    return ax
