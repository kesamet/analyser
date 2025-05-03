"""
Plots.
"""

import altair as alt
import pandas as pd
import plotly.graph_objects as go

from analyser.data import compute_bbands, compute_sma, rebase


def plot_data(
    df: pd.DataFrame,
    title: str = "",
    xlabel: str = "Date",
    ylabel: str = "Price",
    ax=None,
):
    """Plot stock prices."""
    ax = df.plot(title=title, fontsize=12, ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax


def plot_normalized_data(
    df: pd.DataFrame,
    title: str = "",
    xlabel: str = "Date",
    ylabel: str = "Normalized",
    ax=None,
):
    """Plot normalized stock prices."""
    normdf = rebase(df)
    ax = plot_data(normdf, title=title, xlabel=xlabel, ylabel=ylabel, ax=ax)
    ax.axhline(y=1, linestyle="--", color="k")
    return ax


def plot_bollinger(df: pd.DataFrame, title: str = "", ax=None):
    """Plot bollinger bands and SMA."""
    df2 = df[["close"]].copy()
    _, df2["upper"], df2["lower"] = compute_bbands(df["close"])
    df2["sma200"] = compute_sma(df["close"], 200)
    df2["sma50"] = compute_sma(df["close"], 50)

    ax = plot_data(df2, title=title, ax=ax)
    return df2, ax


def plot_with_two_scales(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    xlabel: str = "Date",
    ylabel1: str = "Normalized",
    ylabel2: str | None = None,
):
    """Plot two graphs together."""
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(9, 6.5))

    color = "tab:blue"
    df1.plot(ax=ax1)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel1, color=color)
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = "tab:red"
    df2.plot(ax=ax2, color=color, legend=None)
    ax2.set_ylabel(ylabel2, color=color)
    ax2.tick_params(axis="y", labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


def barchart(source: pd.DataFrame, title: str = "", timeunits: str = "yearmonth"):
    """
    Plot barchart. source must be a Dataframe with only 1 column
    and Date:T as index.
    """
    source = source.reset_index()
    source.columns = ["Date", "Value"]
    return (
        alt.Chart(source, title=title)
        .mark_bar()
        .encode(
            x=alt.X(f"{timeunits}(Date):O", title="Date"),
            y="Value:Q",
            tooltip=[
                alt.Tooltip(f"{timeunits}(Date)", title="Date"),
                alt.Tooltip("Value", title=title),
            ],
        )
    )


def py_ringchart(values: list, labels: list, colors: list, title: str | None = None):
    """Ring chart."""
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines

    fig, ax = plt.subplots()
    fig.set_size_inches(4, 4)
    ax.axis("equal")
    width = 0.25

    pie, _ = ax.pie(values[::-1], radius=1, colors=colors[::-1], startangle=90)
    plt.setp(pie, width=width, edgecolor="white")

    # setting up the legend
    bars = list()
    for label, color in zip(labels, colors):
        bars.append(
            mlines.Line2D(
                [],
                [],
                color=color,
                marker="s",
                linestyle="None",
                markersize=10,
                label=label,
            )
        )

    ax.legend(handles=bars, prop={"size": 8}, loc="center", frameon=False)

    if title is not None:
        ax.set_title(title)
    return fig


def plotly_ringchart(values: list, labels: list, title: str = ""):
    """Plotly ring chart."""
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.5)])
    fig.update_layout(title=title)
    return fig


def plotly_sunburst(names: list, parents: list, values: list, title: str = ""):
    import plotly.express as px

    data = dict(
        names=names,
        parents=parents,
        values=values,
    )
    fig = px.sunburst(
        data,
        names="names",
        parents="parents",
        values="values",
    )
    fig.update_layout(title=title)
    return fig


def histogram_chart(source):
    """Histogram chart."""
    base = alt.Chart(source)
    chart = (
        base.mark_area(
            opacity=0.5,
            interpolate="step",
        )
        .encode(
            alt.X("Prediction:Q", bin=alt.Bin(maxbins=10), title="Prediction"),
            alt.Y("count()", stack=None),
        )
        .properties(
            width=280,
            height=200,
        )
    )
    return chart
