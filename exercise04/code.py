import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

url = "pca_iris.data"
df = pd.read_csv(url, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target'])

x = df[['sepal_length','sepal_width','petal_length','petal_width']]
y = df[['target']]

x = StandardScaler().fit_transform(x)

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

final_df = pd.concat([principalDf, df[['target']]], axis = 1)

dfSetosa = final_df[final_df.target == 'Iris-setosa']
dfVersicolor = final_df[final_df.target == 'Iris-versicolor']
dfVirginica = final_df[final_df.target == 'Iris-virginica']

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

plt.scatter(dfSetosa['principal component 1'], dfSetosa['principal component 2'], color='r', label='Setosa')
plt.scatter(dfVersicolor['principal component 1'], dfVersicolor['principal component 2'], color='b', label='Versicolor')
plt.scatter(dfVirginica['principal component 1'], dfVirginica['principal component 2'], color='g', label='Virginica')


targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']

for target, col in zip(targets, colors):
    dftemp = final_df[final_df['target'] == target]
    plt.scatter(dftemp['principal component 1'], dftemp['principal component 2'], color=col, label=target)

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()

pca.explained_variance_ratio_.sum()
