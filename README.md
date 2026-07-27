# analyser

## 🔧 Getting Started

### Download data
Stock data is download from Yahoo Finance. The symbols of the stocks of interest are first added to the file `symbols.py`.

The data can then be downloaded by
```bash
uv run download.py
```

<details><summary>Shiller data</summary>
<p>

```bash
wget http://www.econ.yale.edu/~shiller/data/ie_data.xls -P ./data/summary
```

</p>
</details>


### Run the Streamlit app

```bash
# Streamlit app
uv run streamlit run app_analyser.py

# reflex app
reflex run
```

## Notebooks
- FRED

  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kesamet/analyser/blob/master/notebooks/test_fred.ipynb)
