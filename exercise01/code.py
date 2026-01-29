import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('diabetes.csv')

df.shape
df.dtypes

seker_hastaleri = df[df.Outcome == 1]
saglikli_hastalar = df[df.Outcome == 0]

# Visualiztion
plt.scatter(saglikli_hastalar.Age, saglikli_hastalar.Glucose, color = "green", label = "Sağlıklılar")
plt.scatter(seker_hastaleri.Age, seker_hastaleri.Glucose, color = "red", label = "Hastalar")
plt.xlabel("yaş")
plt.ylabel("Glikoz")
plt.legend()
plt.show()

y = df.Outcome.values
x_ham_veri = df.drop(["Outcome"], axis=1)

x = ((x_ham_veri - np.min(x_ham_veri)) / (np.max(x_ham_veri) - np.min(x_ham_veri)))

x_train, x_test, y_train, y_test = train_test_split(x,  y, test_size=0.1, random_state=1)

counter = 1
for k in range(1,11):
    knn_yeni = KNeighborsClassifier(n_neighbors=k)
    knn_yeni.fit(x_train, y_train)
    print(counter, ". Doğruluk oranı: ", knn_yeni.score(x_test, y_test))
    counter += 1

sc = MinMaxScaler()
sc.fit_transform(x_ham_veri)

new_prediction = knn.predict(sc.transform(np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])))
new_prediction[0]
