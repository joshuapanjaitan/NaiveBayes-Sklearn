import csv
import numpy as np
import sklearn
from sklearn import naive_bayes
from sklearn.naive_bayes import CategoricalNB


def read(filename):
    data = []
    with open(filename+'.csv', 'r') as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
            data.append(row)

    return data


# baca data ke array,
dataset = read('dataset')
labelSet = []
datatest = read('tes')

for i in range(len(dataset)):
    labelSet.append(dataset[i][4])
    dataset[i].pop(4)
    if i == 0:
        dataset[i][0] = 2

# ganti format array ke array numpy
DataSet = np.array(dataset)
LabelSet = np.array(labelSet)
DataTest = np.array(datatest)

# ganti format isi ke integer
DataSet = DataSet.astype('int32')
LabelSet = LabelSet.astype('int32')
DataTest = DataTest.astype('int32')

goldLabel = np.array([0, 0, 0, 0, 0])  # label Asli


# fungsi library bayes
clf = CategoricalNB()
clf.fit(DataSet, LabelSet)
Prediksi = clf.predict(DataTest)
Akurasi = clf.score(DataTest, goldLabel)
print("Hasil Prediksi = ", Prediksi)
print("Akurasi = ", Akurasi)
