import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from Klasifikator import Klasifikator

def razdeliNaDele(df):

    #Premešam podatke v df
    pomesaniPodatki = df.sample(frac=1, random_state=250)

    steviloDelov = 10

    razdeljeniPodatki = np.array_split(pomesaniPodatki, steviloDelov)

    for i, part in enumerate(razdeljeniPodatki):
        part.to_csv(f"data/cross_validation/part{i}.csv", index=False)


def ustvariGraf(dataframe, vrsta):
    for nacin, group in dataframe.groupby('nacin'):
        plt.plot(group['steviloSosedov'], group['accuracy'], marker='o', linestyle='-', label=nacin)
    plt.xlabel('Stevilo Sosedov')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy/Stevilo Sosedov za Nacin + {vrsta}')
    plt.legend()
    plt.show()
def dataframeToCSV(dataframe, csvSosedje, csvNacin, csvAccuracy, tip):
        dataframe["steviloSosedov"] = csvSosedje
        dataframe["nacin"] = csvNacin
        dataframe["accuracy"] = csvAccuracy
        dataframe.to_csv(f"data/navzkriznaValidacija{tip}.csv", index=False)

def navzkriznaValidacija(mojKlasifikator, df):

    #Razdelimo na dele
    razdeliNaDele(df)

    #Shranimo vse dele v list
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

    #Gremo skoz vsaki tip
    for type in ["evklidska", "manhattan"]:

        #Gremo skoz vsako stevilo sosedov
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

                ############### Za normalizacijo
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

            def povprecniAccuracy(skupenAccuracy, csvSosedje, csvNacin, type):
                print(
                    f"Skupen accuracy za stevilo sosedov {steviloSosedov} z {type} razdaljo: {sum(skupenAccuracy) / len(skupenAccuracy)}")
                csvSosedje.append(steviloSosedov)
                csvNacin.append(type)
                csvAccuracy.append(sum(skupenAccuracy) / len(skupenAccuracy))


            #povprecniAccuracy(skupenAccuracy, csvSosedje, csvNacin, type)
            #povprecniAccuracy(skupenAccuracyNorm, csvSosedjeNorm, csvNacinNorm, type)

            print(f"Skupen accuracy za stevilo sosedov {steviloSosedov} z {type} razdaljo: {sum(skupenAccuracy) / len(skupenAccuracy)}")
            csvSosedje.append(steviloSosedov)
            csvNacin.append(type)
            csvAccuracy.append(sum(skupenAccuracy) / len(skupenAccuracy))

            print(f"Skupen accuracy za stevilo sosedov {steviloSosedov} z {type} razdaljo in normalizacijo: {sum(skupenAccuracyNorm) / len(skupenAccuracyNorm)}")
            csvSosedjeNorm.append(steviloSosedov)
            csvNacinNorm.append(type)
            csvAccuracyNorm.append(sum(skupenAccuracyNorm) / len(skupenAccuracyNorm))




    #Shranimo v csv
    dataframeToCSV(csvDataframe, csvSosedje, csvNacin, csvAccuracy, "basic")
    dataframeToCSV(csvDataframeNorm, csvSosedjeNorm, csvNacinNorm, csvAccuracyNorm, "normalizacija")

    #Ustvarimo grafe
    ustvariGraf(csvDataframe, "brez normalizacije")
    ustvariGraf(csvDataframeNorm, "normalizacija")


#Preberemo podatke ter jih shranimo v df
df = pd.read_csv('data/IRIS.csv')

#Ustvarimo razred za klasifikacijo
mojKlasifikator = Klasifikator(5, 'manhattan')

#Razdelitev na testno in učno množico, Random state je seed
train = df.sample(frac=0.8, random_state=200)

test = df.drop(train.index)

#test.to_csv("data/test.csv", index=False)
mojKlasifikator.fit(train)

testna = mojKlasifikator.predictBasic(test)
testna.to_csv("data/klasificiran.csv", index=False)

print(f"Točnost te množice je: {mojKlasifikator.test(testna)} ")

### Precna validacija
navzkriznaValidacija(mojKlasifikator, df)






















