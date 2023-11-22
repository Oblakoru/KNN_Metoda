import math
import os
import pandas as pd
from scipy.spatial.distance import cityblock
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import random


class Klasifikator:

    def __init__(self):
        pass

    # Konstrutktor
    def __init__(self, steviloSosedov, nacinIzracuna):
        self.steviloSosedov = steviloSosedov

        if nacinIzracuna == "evklidska" or nacinIzracuna == "manhattan":
            self.nacinIzracuna = nacinIzracuna
        else:
            print("Napačen način izračuna, uporabljen bo evklidski")
            self.nacinIzracuna = "evklidska"

        self.df_ucna_mnozica = None
        self.df_testna_mnozica = None


    # Nastavim učno množico
    def fit(self, ucnaMnozica):
        self.df_ucna_mnozica = ucnaMnozica


    # Predikcija za testno množico
    def predictBasic(self, testnaMnozica):

        # List za shranjevanje predikcij - vrsta rastline
        predikcijeList = []

        # Sprehod skozi vse vrstice v testni množici
        for rowTest in testnaMnozica.values:

            # List za shranjevanje razdalj
            razdalje = []

            # ZA vsako vrsto v učni množici izračunam razdaljo do testne instance
            if self.nacinIzracuna == "evklidska":
                for row in self.df_ucna_mnozica.values:
                    distance = math.dist(rowTest[:-1], row[:-1])
                    razdalje.append(distance)
            else:
                for row in self.df_ucna_mnozica.values:
                    distance = cityblock(row[:-1], rowTest[:-1])
                    razdalje.append(distance)

            # Naredim kopijo, da se ne spreminja originalni df
            dfUcnaRazdalje = self.df_ucna_mnozica.copy()

            # Dodam stolpec z razdaljami
            dfUcnaRazdalje["Razdalja"] = razdalje

            # Sortiram po razdalji in vzamem prvih x
            ocene = dfUcnaRazdalje.sort_values(by="Razdalja", ascending=True, inplace=False).head(self.steviloSosedov)[
                "species"]

            # #Damo v dictionary
            # oceneDictionary = (ocene.value_counts().to_dict())
            #
            # #Pregledamo, koliko je max ponavljanje elementov
            # max_occurrences = max(oceneDictionary.values())
            #
            # # Najdemo vse elemente, ko se največkrat ponavlajo
            # max_occurrence_names = [name for name, count in oceneDictionary.items() if count == max_occurrences]
            #
            # #print(max_occurrence_names)
            #
            # #Preverimo, če jih je več
            # if len(max_occurrence_names) > 1:
            #     # Zbere eno random, če jih je več z isto vrednostjo
            #     print(max_occurrence_names)
            #     print(oceneDictionary)
            #     print(ocene)
            #     print(f"Več enakih pri {self.steviloSosedov} - sledi random zbiranje")
            #     ocena = random.choice(max_occurrence_names)
            #     print("Izbrana ocena: " + ocena)
            #     predikcijeList.append(ocena)
            #     #print(f"Klasificiran kot: {ocena}, dejansko: {rowTest[-1]}")
            # else:
            #     predikcijeList.append(max_occurrence_names[0])
            #     #print(f"Klasificiran kot: {max_occurrence_names[0]}, dejansko: {rowTest[-1]}")

            # Dodam v seznam, kjer so vsi rezultati
            predikcijeList.append(ocene.value_counts().idxmax())

        testnaMnozica["predikcija"] = predikcijeList
        #print(self.test(testnaMnozica))

        return testnaMnozica

    ##Testiranje
    def test(self, data):

        y_true = data['species']
        y_pred = data['predikcija']

        accuracy = accuracy_score(y_true, y_pred)
        #print(f'Natancnost poskusa: {accuracy:.2f}')

        return accuracy


