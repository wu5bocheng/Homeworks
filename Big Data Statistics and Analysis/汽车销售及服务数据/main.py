#coding = utf-8
import numpy as np
import pandas as pd
data = pd.read_csv("car_info_train.csv",encoding="UTF-8")
usage = data["CAR_PRICE"]
print(usage)