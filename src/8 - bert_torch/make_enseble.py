import numpy as np
import os

predictions = ["path/to/checkpoint0", "path/to/checkpoint1", "..."]

data = []

for pred in predictions:
    with open(os.path.join(pred, "prediction.npy"), "rb") as f:
        a = np.load(f)
        data.append(a)

p = np.asarray(data)
mean = np.mean(p, axis=0)

y_preds = np.argmax(mean, axis=1)

y_preds = [-1 if val == 0 else 1 for val in y_preds]

import pandas as pd
df = pd.DataFrame(y_preds, columns=["Prediction"])
df.index.name = "Id"
df.index += 1
df.to_csv("test_data_ensemble.csv")

exit(0)
