# analyser

## 🔧 Getting Started

You will need to set up your development environment using conda, which you can install [directly](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).
```bash
conda env create --name analyser python=3.11
```

Activate the environment and install the Python packages.
```bash
conda activate analyser
pip install -r requirements.txt
```


### Download data
Stock data is download from Yahoo Finance. The symbols of the stocks of interest are first added to the file `symbols.py`.

The data can then be downloaded by
```bash
python -m download
```

<details><summary>Shiller data</summary>
<p>

```bash
wget http://www.econ.yale.edu/~shiller/data/ie_data.xls -P ./data/summary
```

</p>
</details>


## 💻 App

To run Streamlit app,
```bash
streamlit run app_analyser.py
```


## Notebooks
- FRED[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kesamet/analyser/blob/master/notebooks/test_fred.ipynb)
