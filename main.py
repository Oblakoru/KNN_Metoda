import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from Klasifikator import Klasifikator

def razdeliNaDele():
    #Premešam podatke v df
    pomesaniPodatki = df.sample(frac=1, random_state=200)

    steviloDelov = 10

    razdeljeniPodatki = np.array_split(pomesaniPodatki, steviloDelov)

    for i, part in enumerate(razdeljeniPodatki):
        part.to_csv(f"data/cross_validation/part{i}.csv", index=False)


def navzkriznaValidacija(mojKlasifikator):
    folder_path = "data/cross_validation"
    files = os.listdir(folder_path)

    csvDataframe = pd.DataFrame()
    csvSosedje = []
    csvNacin = []
    csvAccuracy = []

    csvDataframeNorm = pd.DataFrame()
    csvSosedjeNorm = []
    csvNacinNorm = []
    csvAccuracyNorm = []

    for type in ["evklidska", "manhattan"]:

        for steviloSosedov in [1, 2, 3, 4, 5, 6, 7, 8]:

            skupenAccuracy = []
            skupenAccuracyNorm = []

            for i in range(len(files)):

                testniPodatek = files[i]
                ucniPodatek = [f for j, f in enumerate(files) if j != i]

                testniPodatekDF = pd.read_csv(f"data/cross_validation/{testniPodatek}")
                ucniPodatekDF = pd.concat((pd.read_csv(f"data/cross_validation/{f}") for f in ucniPodatek),
                                          ignore_index=True)

                mojKlasifikator.fit(ucniPodatekDF)
                mojKlasifikator.steviloSosedov = steviloSosedov
                mojKlasifikator.nacinIzracuna = type

                zaAccuracy = mojKlasifikator.predictBasic(testniPodatekDF)

                skupenAccuracy.append(mojKlasifikator.test(zaAccuracy))


                ###### Z normalizacijo


                for column in ucniPodatekDF.columns:
                    if column != "species":
                        ucniPodatekDF[column] = (ucniPodatekDF[column] - ucniPodatekDF[column].min()) / (
                                    ucniPodatekDF[column].max() - ucniPodatekDF[column].min())

                testniPodatekDF.drop("predikcija", axis=1, inplace=True)
                for column in testniPodatekDF.columns:
                    if column != "species":
                        testniPodatekDF[column] = (testniPodatekDF[column] - testniPodatekDF[column].min()) / (
                                    testniPodatekDF[column].max() - testniPodatekDF[column].min())

                mojKlasifikator.fit(ucniPodatekDF)

                zaAccuracy = mojKlasifikator.predictBasic(testniPodatekDF)

                skupenAccuracyNorm.append(mojKlasifikator.test(zaAccuracy))


            print(f"Skupen accuracy za stevilo sosedov {steviloSosedov} z {type} razdaljo: {sum(skupenAccuracy) / len(skupenAccuracy)}")
            csvSosedje.append(steviloSosedov)
            csvNacin.append(type)
            csvAccuracy.append(sum(skupenAccuracy) / len(skupenAccuracy))

            print(f"Skupen accuracy za stevilo sosedov {steviloSosedov} z {type} razdaljo in normalizacijo: {sum(skupenAccuracyNorm) / len(skupenAccuracyNorm)}")
            csvSosedjeNorm.append(steviloSosedov)
            csvNacinNorm.append(type)
            csvAccuracyNorm.append(sum(skupenAccuracyNorm) / len(skupenAccuracyNorm))


    csvDataframe["steviloSosedov"] = csvSosedje
    csvDataframe["nacin"] = csvNacin
    csvDataframe["accuracy"] = csvAccuracy
    csvDataframe.to_csv("data/navzkriznaValidacija.csv", index=False)

    csvDataframeNorm["steviloSosedov"] = csvSosedjeNorm
    csvDataframeNorm["nacin"] = csvNacinNorm
    csvDataframeNorm["accuracy"] = csvAccuracyNorm
    csvDataframeNorm.to_csv("data/navzkriznaValidacijaNorm.csv", index=False)

    # Group by 'nacin' and plot each group separately
    for nacin, group in csvDataframe.groupby('nacin'):
        plt.plot(group['steviloSosedov'], group['accuracy'], marker='o', linestyle='-', label=nacin)

    plt.xlabel('Stevilo Sosedov')
    plt.ylabel('Accuracy')
    plt.title('Accuracy/Stevilo Sosedov za Nacin')
    plt.legend()
    plt.show()

    for nacin, group in csvDataframeNorm.groupby('nacin'):
        plt.plot(group['steviloSosedov'], group['accuracy'], marker='o', linestyle='-', label=nacin)

    plt.xlabel('Stevilo Sosedov')
    plt.ylabel('Accuracy')
    plt.title('Accuracy/Stevilo Sosedov za Nacin + Normalizacija')
    plt.legend()
    plt.show()


#Preberemo podatke ter jih shranimo v df
df = pd.read_csv('data/IRIS.csv')

#Ustvarimo razred za klasifikacijo
mojKlasifikator = Klasifikator(5, 'evklidska')

#Razdelitev na testno in učno množico, Random state je seed
train = df.sample(frac=0.8, random_state=500)
test = df.drop(train.index)

#test.to_csv("data/test.csv", index=False)

mojKlasifikator.fit(train)

testna = mojKlasifikator.predictBasic(test)

mojKlasifikator.test(testna)

### Precna validacija

navzkriznaValidacija(mojKlasifikator)






















