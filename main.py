import pandas as pd
from Klasifikator import Klasifikator

#Preberemo podatke ter jih shranimo v df
df = pd.read_csv('data/IRIS.csv')

mojKlasifikator = Klasifikator(5, 'evklidska')

#Razdelitev na testno in učno množico
train = df.sample(frac=0.8, random_state=200)
test = df.drop(train.index)

mojKlasifikator.fit(train)

mojKlasifikator.predict(test)

