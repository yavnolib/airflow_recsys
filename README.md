# airflow_recsys
A repository for running recsys models with different parameters with logging experiments in mlflow and adapted for launching using Airflow.

# Install
```
docker pull yavnolib/air_aaa_mlsd
```

# Run:
* mode = 1: run baseline model
* mode = 2: run best model
* mode = 3: fit baseline with optuna
* mode = 4: custom run with your params (see 'help')


# Help:
```
python main.py --help
```
