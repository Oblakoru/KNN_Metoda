import math
import os
import pandas as pd
from scipy.spatial.distance import cityblock
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


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

        # List za shranjevanje predikcij
        oceneList = []

        # Sprehod skozi vse vrstice v testni množici
        for rowTest in testnaMnozica.values:

            # List za shranjevanje razdalj
            razdalje = []

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

            #print(f"Klasificiran kot: {ocene.value_counts().idxmax()}, dejansko: {rowTest[-1]}")

            # Dodam v seznam, kjer so vsi rezultati
            oceneList.append(ocene.value_counts().idxmax())

        testnaMnozica[f"predikcija"] = oceneList

        #testnaMnozica.to_csv("data/klasifikacija.csv", index=False)

        return testnaMnozica


    ##Testiranje
    def test(self, data):

        y_true = data['species']
        y_pred = data['predikcija']

        accuracy = accuracy_score(y_true, y_pred)
        #print(f'Natancnost poskus: {accuracy:.2f}')

        return accuracy


    # def predictZaNavzkrizno(self, testnaMnozica, ucnaMnozica, steviloSosedov, nacinIzracuna):
    #
    #     oceneList = []
    #
    #     for rowTest in testnaMnozica.values:
    #         razdalje = []
    #         if nacinIzracuna == "evklidska":
    #             for row in ucnaMnozica.values:
    #                 distance = math.dist(rowTest[:-1], row[:-1])
    #                 razdalje.append(distance)
    #         else:
    #             for row in ucnaMnozica.values:
    #                 distance = cityblock(row[:-1], rowTest[:-1])
    #                 razdalje.append(distance)
    #
    #         # Naredim kopijo, da se ne spreminja originalni df
    #         dfRazdalje = ucnaMnozica.copy()
    #
    #         # Dodam stolpec z razdaljami
    #         dfRazdalje["Razdalja"] = razdalje
    #
    #         # Sortiram po razdalji in vzamem prvih x
    #         ocene = dfRazdalje.sort_values(by="Razdalja", inplace=False).head(steviloSosedov)["species"]
    #
    #         # print(f"Klasificiran kot: {ocene.value_counts().idxmax()}, dejansko: {rowTest[-1]}")
    #
    #         # Dodam v seznam, kjer so vsi rezultati
    #         oceneList.append(ocene.value_counts().idxmax())
    #
    #     testnaMnozica["predikcija"] = oceneList
    #
    #     return testnaMnozica


    # def navzkriznaValidacija(self):
    #
    #     folder_path = "data/cross_validation"
    #     files = os.listdir(folder_path)
    #
    #     for type in ["evklidska", "manhattan"]:
    #
    #         for steviloSosedov in [1, 2, 3, 4, 5, 6, 7, 8]:
    #
    #             skupenAccuracy = []
    #
    #             for i in range(len(files)):
    #                 testniPodatek = files[i]
    #                 ucniPodatek = [f for j, f in enumerate(files) if j != i]
    #
    #                 testniPodatekDF = pd.read_csv(f"data/cross_validation/{testniPodatek}")
    #                 ucniPodatekDF = pd.concat((pd.read_csv(f"data/cross_validation/{f}") for f in ucniPodatek),
    #                                           ignore_index=True)
    #
    #                 zaAccuracy = self.predictZaNavzkrizno(testniPodatekDF, ucniPodatekDF, steviloSosedov, type)
    #
    #                 skupenAccuracy.append(self.test(zaAccuracy))
    #
    #             print(
    #                 f"Skupen accuracy za stevilo sosedov {steviloSosedov} z {type} razdaljo: {sum(skupenAccuracy) / len(skupenAccuracy)}")
