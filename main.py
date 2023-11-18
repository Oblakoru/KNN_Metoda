import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from Klasifikator import Klasifikator

#Preberemo podatke ter jih shranimo v df
df = pd.read_csv('data/IRIS.csv')

mojKlasifikator = Klasifikator(5, 'evklidska')

#Razdelitev na testno in učno množico

#Od random_state je odvisno, kako se bodo podatki razdelili - ce je isti so zmeraj enako razdeljeni
train = df.sample(frac=0.8, random_state=200)
#train.to_csv("data/train.csv", index=False)

test = df.drop(train.index)
#test.to_csv("data/test.csv", index=False)

mojKlasifikator.fit(train)

mojKlasifikator.predictBasic(test)



#####Cross validation



def razdeliNaDele():
    #Premešam podatke v df
    pomesaniPodatki = df.sample(frac=1, random_state=200)

    steviloDelov = 10

    razdeljeniPodatki = np.array_split(pomesaniPodatki, steviloDelov)

    # Display each part
    for i, part in enumerate(razdeljeniPodatki):
        part.to_csv(f"data/cross_validation/part{i}.csv", index=False)











