import math
import pandas as pd
from scipy.spatial.distance import cityblock
from Klasifikator import Klasifikator

#Preberemo podatke ter jih shranimo v df
df = pd.read_csv('data/IRIS.csv')

mojKlasifikator = Klasifikator(3, 'evklidska')

train = df.sample(frac=0.8, random_state=200)
test = df.drop(train.index)

print("Train: ", train)
print("Test: ", test)

mojKlasifikator.fit(train.values)

