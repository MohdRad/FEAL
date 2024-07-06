# README

1. Code for a possible publication about using Active Learning to discover the free energy of enouromous number of Polycyclic Aromatic Carbons (PACs) using limited number of samples.
2. [`modAL`](https://modal-python.readthedocs.io/en/latest/) is an active learning package. Installing it on anaconda using pip has issues so we include it here.
3. There are 313 input features to predict free energy of dimerization, last column in data_all.csv (assoc).

## Installation

To set up the environment and install the requirements, run the following command in in the root directory of the repository using the terminal:

```bash
conda create --prefix ./feal python=3.10 --yes
conda activate ./feal
conda config --set env_prompt '(feal)'
pip install -r requirements.txt
```

## Usage

To run the code, use the following command:

```bash
python feal.py
```

This function uses precomputed csv files to indicate the training data. ==How are these files generated? Why do we need different files if the same dataset is used, just with different training and testing sets?== It compares active learning and random learning on several cases:

1. **The original feature space using all data.** This may lead to bias because no PAC is actually held out, just the interactions between PACs.
2. **The original feature space with holdout PACs.** This shows a realistic scenario where we want to apply to model to new PACs that were not seen at training time.
3. **The reduced feature space using all data.** This evaluates the impact of a reduced feature space, which saves computational resources.
4. **The reduced feature space with holdout PACs.** This is the most realistic scenario, where we want to apply the model to new PACs that were not seen at training time and we use a reduced feature space.

==The code should not be commented out. It should be handled more elegantly. Possibly using commandline arguments via [argparse](https://docs.python.org/3/library/argparse.html).==


