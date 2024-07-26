# Detecting Routines from nursery home residents

## Install dependencies

Install all the dependencies in the requirements.txt file.

```bash
cd setup
pip install -r requirements.txt
```

Or create a new virtual environment with the dependencies.

```bash
cd setup
conda env create --name <env_name> --file = config.yml
```

## Run the routine execution

To run the routine execution, run the following command:

```bash
python main.py
```

This code will save on the `results` file the results of the routine execution.

The parameters, directory of data and the name of the results directory are on the `config.yaml` file.

## Run the frequency table extraction

To run the frequency table extraction from the results of the routine execution, run the following command:

```bash
python evaluation.py
```


## Run the metrics calculation

To run the table calculation of ROC AUC, F1 Score, Precision, Recall, ROC plots and confusion matrix from the results of the routine execution, run the following command:

```bash
python metrics.py
```



