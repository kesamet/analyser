# analyser

## 🔧 Getting Started

You will need to set up your development environment using conda, which you can install [directly](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

```bash
conda env create --name analyser -f environment.yaml --force
```

Activate the environment.
```bash
conda activate analyser
```


### Download data
Data is download from Yahoo Finance. The stocks of interest are added to the file `symbols.py`.

To download data,
```bash
python -m download
```


## 💻 App

To run Streamlit app,
```bash
streamlit run app_analyser.py
```
