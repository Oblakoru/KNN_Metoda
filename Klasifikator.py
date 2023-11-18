import math
from sklearn.model_selection import train_test_split

from scipy.spatial.distance import cityblock


class Klasifikator():
    dataframeRazdalj = None
    new_instance = [5.0, 3.0, 1.3, 0.2]

    vrednostiAtributov = None

    def __init__(self, steviloSosedov, nacinIzracuna="evklidska"):
        self.steviloSosedov = steviloSosedov
        self.nacinIzracuna = nacinIzracuna






    def fit(self, vrednostiAtributov):
        print("Klasifikator je pripravljen na klasifikacijo.")
        self.vrednostiAtributov = vrednostiAtributov

        #train, test = train_test_split(vrednostiAtributov, test_size=0.2)



    def predict(self):
        print("Klasifikator je klasificiral instanco.")
        razdalje = []

        if self.nacinIzracuna == "evklidska":

            for row in self.vrednostiAtributov:
                distance = math.dist(self.new_instance, row[:-1])
                razdalje.append(distance)
        else:

            for row in self.vrednostiAtributov:
                distance = cityblock(row[:-1], self.new_instance)
                razdalje.append(distance)

        self.dataframeRazdalj = razdalje

        print(self.dataframeRazdalj)


