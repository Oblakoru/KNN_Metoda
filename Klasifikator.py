import math
import os

import pandas as pd
from scipy.spatial.distance import cityblock
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


class Klasifikator():

    def __init__(self, steviloSosedov, nacinIzracuna="evklidska"):
        self.steviloSosedov = steviloSosedov
        self.nacinIzracuna = nacinIzracuna
        self.df_ucna_mnozica = None
        self.df_testna_mnozica = None


    def fit(self, x):
        self.df_ucna_mnozica = x


    def predict(self, testnaMnozica):

        self.df_testna_mnozica = testnaMnozica

        testnaMnozicaKopija = testnaMnozica.copy()

        for x in [1, 2, 3, 4, 5, 6, 7, 8]:

            oceneList = []

            print(self.df_ucna_mnozica)



            for rowTest in self.df_testna_mnozica.values:

                razdalje = []

                if self.nacinIzracuna == "evklidska":

                    for row in self.df_ucna_mnozica.values:
                        distance = math.dist(rowTest[:-1], row[:-1])
                        razdalje.append(distance)
                else:

                    for row in self.df_ucna_mnozica.values:
                        distance = cityblock(row[:-1], rowTest[:-1])
                        razdalje.append(distance)

                dfRazdalje = self.df_ucna_mnozica.copy()
                dfRazdalje["Razdalja"] = razdalje

                ocene = dfRazdalje.sort_values(by="Razdalja", inplace=False).head(x)["species"]

                print(f"Klasificiran kot: {ocene.value_counts().idxmax()}, dejansko: {rowTest[-1]}")
                oceneList.append(ocene.value_counts().idxmax())

            testnaMnozicaKopija[f"Klasifikacija{x}"] = oceneList

        testnaMnozicaKopija.to_csv("data/klasifikacija.csv", index=False)

    def predictBasic(self, testnaMnozica):

        self.df_testna_mnozica = testnaMnozica

        oceneList = []

        for rowTest in self.df_testna_mnozica.values:
            razdalje = []
            if self.nacinIzracuna == "evklidska":
                for row in self.df_ucna_mnozica.values:
                    distance = math.dist(rowTest[:-1], row[:-1])
                    razdalje.append(distance)
            else:
                for row in self.df_ucna_mnozica.values:
                    distance = cityblock(row[:-1], rowTest[:-1])
                    razdalje.append(distance)


            #Naredim kopijo, da se ne spreminja originalni df
            dfRazdalje = self.df_ucna_mnozica.copy()

            #Dodam stolpec z razdaljami
            dfRazdalje["Razdalja"] = razdalje

            #Sortiram po razdalji in vzamem prvih x
            ocene = dfRazdalje.sort_values(by="Razdalja", inplace=False).head(self.steviloSosedov)["species"]

            print(f"Klasificiran kot: {ocene.value_counts().idxmax()}, dejansko: {rowTest[-1]}")

            #Dodam v seznam, kjer so vsi rezultati
            oceneList.append(ocene.value_counts().idxmax())

        testnaMnozica[f"predikcija"] = oceneList
        testnaMnozica.to_csv("data/klasifikacija.csv", index=False)

    ##Testiranje
    def test(self):
        df = pd.read_csv('data/klasifikacija.csv')

        y_true = df['species']
        y_pred = df['predikcija']

        accuracy = accuracy_score(y_true, y_pred)
        print(f'Accuracy: {accuracy:.2f}')

        #conf_matrix = confusion_matrix(y_true, y_pred)
        #print('Confusion Matrix:')
        #print(conf_matrix)
        #
        #class_report = classification_report(y_true, y_pred)
        #print('Classification Report:')
        #print(class_report)

    def navzkriznaValidacija(self):

        folder_path = "data/cross_validation"  # Change this to the path of your folder
        files = os.listdir(folder_path)

        for i in range(len(files)):
            testniPodatek = files[i]
            ucniPodatek = [f for j, f in enumerate(files) if j != i]

            testniPodatekDF = pd.read_csv(f"data/cross_validation/{testniPodatek}")
            ucniPodatekDF = pd.concat((pd.read_csv(f"data/cross_validation/{f}") for f in ucniPodatek),
                                      ignore_index=True)





