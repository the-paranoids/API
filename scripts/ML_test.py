import pandas as pd
import numpy as np
from sklearn import svm

data = pd.DataFrame.from_csv("../data/DSL-StrongPasswordData.csv")
print(data)
subjects = data['sessionIndex']
print(subjects)
