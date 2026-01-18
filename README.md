# analyser

## 🔧 Getting Started

### 1. Set up a virtual environment

**conda**
```bash
conda env create --name analyser python=3.12
conda activate analyser
pip install -r requirements.txt
```

**venv**
```bash
# On Windows:
python -m venv .venv
.venv\Scripts\activate

# On macOS/Linux:
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Download data
Stock data is download from Yahoo Finance. The symbols of the stocks of interest are first added to the file `symbols.py`.

The data can then be downloaded by
```bash
python -m download

# or

uv run download.py
```

<details><summary>Shiller data</summary>
<p>

```bash
wget http://www.econ.yale.edu/~shiller/data/ie_data.xls -P ./data/summary
```

</p>
</details>


### 3. Run the app

```bash
# Streamlit app
streamlit run app_analyser.py

# reflex app
reflex run
```


## Notebooks
- FRED

  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kesamet/analyser/blob/master/notebooks/test_fred.ipynb)
