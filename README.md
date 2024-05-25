# README

1. Code for a possible publication about using Active Learning to discover the free energy of enouromous number of Polycyclic Aromatic Carbons (PACs) using limited number of samples.
2. [`modAL`](https://modal-python.readthedocs.io/en/latest/) is an active learning package. Installing it on anaconda using pip has issues so we include it here.
3. There are 311 input features to predict free energy of dimerization, last column in data_all.csv (assoc).

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

## Matt notes

### `FEAL.py`

- You should set the shuffle seed so it's deterministic.
- Why are there different datasets for the PCA and non-PCA cases?
- You are fitting the MinMaxScalar to both the training and validation data. This will negatively impact the validation performance. You should fit the MinMaxScalar to the training data and transform the validation data without refitting.
- Ideally you should not use global variables. You should pass the variables as arguments to the functions and return the modified versions. Think about either using a functional or an object-oriented approach.
- Seems like `joblib.dump(regressor, "model.pkl")` will overwrite the model during each iteration of the for loop. You should probably use a different filename for each iteration and for each combination of dataset/PCA so that you can reference the old model instead of rerunning the whole process each time. You should also put checks in the for loop that load the model from that iteration if it exists.
- Ideally you wold have the functions `FEAL` and `rand` implemented as one function and pass different sampling strategies to them to ensure that the process is the same for both and to reduce code duplication.
- In `rand` your query strategy isn't random, it's to get the next sample in the dataset. This could be random, but you don't shuffle the dataset first. Why do you think this would be a random query strategy? How did you randomize the dataset rows?
- You have some seemingly superfluous assignments. For example, `y_true = y_val` seems to do nothing but change the name of the variable. You should remove these to make the code more readable.
- I think your Committee regressor doesn't work because you're using the same gaussian process model with the same random seed and same hyperparameters for each active learner in the committee. You should at least have a different initial state for each learner. Otherwise, the committee will give the same output as a single model.

### `DR.py`

- You do not provide a random state for PCA, SVD, or ICA.

### General comments

- You have one test where you plot 100 repetitions of the active learner. This implementation seems inefficient because you're refitting the model from scratch every time you want to collect more samples (30, ..., 100). It seems like you should cache the results along the way to 100 samples and plot the model's intermediate output as you add more features. This would probably cut your runtime from n^2 to n.
