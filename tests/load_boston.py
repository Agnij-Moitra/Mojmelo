import pandas as pd

def get_data():
    data = pd.read_csv(r"/home/snape/projects/Mojmelo/tests/BostonHousing.csv")

    return [data.iloc[:, :-1].values, data.iloc[:, -1].values.reshape(-1,1)]
