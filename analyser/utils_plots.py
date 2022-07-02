"""
Utility functions for plots.
"""
import altair as alt
import pandas as pd


def make_source_waterfall(shap_exp, max_display=10):
    """Prepare dataframe for waterfall chart."""
    base_value = shap_exp.base_values
    df = pd.DataFrame(
        {
            "feature": shap_exp.feature_names,
            "feature_value": shap_exp.data,
            "shap_value": shap_exp.values,
        }
    )

    df["abs_val"] = df["shap_value"].abs()
    df = df.sort_values("abs_val", ascending=True)

    df["val_"] = df["shap_value"].values
    remaining = df["shap_value"].iloc[:-max_display].sum()
    output_value = df["shap_value"].sum() + base_value

    df0 = pd.DataFrame(
        {
            "feature": ["Average Model Output"],
            "shap_value": [base_value],
            "val_": [base_value],
        }
    )
    df2 = pd.DataFrame(
        {
            "feature": [f"{len(df) - max_display} other features"],
            "shap_value": [remaining],
            "val_": [remaining],
        }
    )
    df1 = df.iloc[-max_display:]
    df4 = pd.DataFrame(
        {
            "feature": ["Individual Observation"],
            "shap_value": [output_value],
            "val_": [0],
        }
    )
    source = pd.concat([df0, df2, df1, df4], axis=0, ignore_index=True)

    source["close"] = source["val_"].cumsum()
    source["open"] = source["close"].shift(1)
    source.loc[len(source) - 1, "open"] = 0
    source["open"].fillna(0, inplace=True)
    return source.iloc[::-1]


def waterfall_chart(source, decimal=3):
    """Waterfall chart."""
    source = source.iloc[1:-1].copy()
    for c in ["feature_value", "shap_value"]:
        source[c] = source[c].round(decimal).astype(str)
    source.loc[source["feature_value"] == "nan", "feature_value"] = ""

    bars = (
        alt.Chart(source)
        .mark_bar()
        .encode(
            alt.Y("feature:O", sort=source["feature"].tolist(), title=""),
            alt.X("open:Q", scale=alt.Scale(zero=False), title=""),
            alt.X2("close:Q"),
            alt.Tooltip(["feature", "feature_value", "shap_value"]),
        )
    )
    color1 = bars.encode(
        color=alt.condition(
            "datum.open <= datum.close",
            alt.value("#FF0D57"),
            alt.value("#1E88E5"),
        ),
    )
    color2 = bars.encode(
        color=alt.condition(
            "datum.feature == 'Average Model Output' || datum.feature == 'Individual Observation'",
            alt.value("#F7E0B6"),
            alt.value(""),
        ),
    )
    return bars + color1 + color2


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
