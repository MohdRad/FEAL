# README

1. Code for a possible publication about using Active Learning to discover the free energy of an enormous number of Polycyclic Aromatic Carbons (PACs) using a limited number of samples.
2. [`modAL`](https://modal-python.readthedocs.io/en/latest/) is an active learning package. Installing it on Anaconda using pip has issues, so we include it here.
3. There are 311 input features to predict free energy of dimerization, last column in data_all.csv (assoc).

## Installation

To set up the environment and install the requirements, run the following command in the root directory of the repository using the terminal:

```bash
conda create --prefix ./feal python=3.10 --yes
conda activate ./feal
conda config --set env_prompt '(feal)'
pip install -r requirements.txt
```

## Usage

To run the code, use the following command:

```bash
python run.py
```
To get the stratified sampling results, run the following:
```bash
python run_stratified.py
```


