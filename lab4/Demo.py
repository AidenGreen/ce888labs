import pandas as pd
import numpy as np
from IPython.display import Image

data = pd.read_csv("jester-data-1.csv")
print(data[0:1])

data["Lable"] = 0
sample = data.sample(10)
sample["Lable"] = 99
print(sample["Lable"])
print(sample.index)