import math
from sklearn.model_selection import train_test_split

from scipy.spatial.distance import cityblock


class Klasifikator():

    #new_instance = [5.0, 3.0, 1.3, 0.2]
    

    df_ucna_mnozica = None
    df_testna_mnozica = None

    def __init__(self, steviloSosedov, nacinIzracuna="evklidska"):
        self.steviloSosedov = steviloSosedov
        self.nacinIzracuna = nacinIzracuna


    def fit(self, vrednostiAtributov):

        print("Klasifikator je pripravljen na klasifikacijo.")
        self.df_ucna_mnozica = vrednostiAtributov

        print(self.df_ucna_mnozica)


    def predict(self, testnaMnozica):

        self.df_testna_mnozica = testnaMnozica

        print(testnaMnozica)
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

            ocene = dfRazdalje.sort_values(by="Razdalja", inplace=False).head(self.steviloSosedov)["species"]

            print(f"Klasificiran kot: {ocene.value_counts().idxmax()}, dejansko: {rowTest[-1]}")






