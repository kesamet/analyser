from datetime import timedelta

import altair as alt
import pandas as pd


def plot_linechart(
    source,
    cutoff=None,
    xtitle="Time",
    xformat=None,
    ytitle="",
    yscale=None,
    tformat="%Y-%m-%d %H:%M",
    crange=None,
    title="",
):
    """Custom line chart."""
    xargs = {"title": xtitle}
    if xformat:
        xargs["axis"] = alt.Axis(format=xformat)

    yargs = {"title": ytitle}
    if yscale:
        yargs["scale"] = alt.Scale(domain=yscale)

    line = alt.Chart(source).mark_line().encode(
        x=alt.X("timestamp:T", **xargs),
        y=alt.Y("value:Q", **yargs),
        tooltip=[
            alt.Tooltip("timestamp:T", title=xtitle, format=tformat),
            alt.Tooltip("value:Q", title=ytitle),
        ],
    ).properties(title=title)

    if "variable" in source.columns:
        cargs = {"title": None}  # , "legend": alt.Legend(orient="bottom")
        if crange:
            cargs["scale"] = alt.Scale(range=crange)
        line = line.encode(color=alt.Color("variable:N", **cargs))

    if cutoff is not None:
        areas = alt.Chart(cutoff).mark_rect(color="black", opacity=0.2).encode(
            x='start',
            x2='stop',
            # y=alt.value(0),  # pixels from top
            # y2=alt.value(300),  # pixels from top
        )
    return line + areas


def get_cutoff(recession, min_date, max_date):
    tmp = recession[recession.index >= min_date]
    starts = list(tmp.query("diff == 1").index)
    stops = [t - timedelta(days=1) for t in tmp.query("diff == -1").index]
    if starts[0] < stops[0]:
        if len(starts) != len(stops):
            stops.append(max_date)
    else:
        if len(starts) != len(stops):
            stops = [min_date] + stops
        else:
            starts = [min_date] + starts
            stops.append(max_date)
    return pd.DataFrame({"start": starts, "stop": stops})
