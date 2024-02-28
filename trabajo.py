import matplotlib.pyplot as plt
import sklearn
import pandas as pd
import mpl_toolkits.mplot3d  # noqa: F401


#importancion de modulos necesarios

from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA
from sklearn import datasets
import numpy as np

#cargamos lo datos 
dataset = datasets.load_breast_cancer()
#print(dataset) 
print("informacion del cojunto de entrada")
print(dataset.keys())
#Verificamos las características del conjunto de datos de entrada 
print('Características del conjunto de datos de entrada:') 
print(dataset.DESCR) 
print("Datos de cancer han finalizado..........................")
cancer = datasets.load_breast_cancer()


#creamos la grafica de barras para el cancer
# Contamos las ocurrencias de cada clase (0 y 1)
class_counts = [sum(cancer.target == 0), sum(cancer.target == 1)]

# Etiquetas de las clases
class_labels = ['Benigno', 'Maligno']

# Creamos la gráfica de barras
plt.bar(class_labels, class_counts, color=['blue', 'red'])

# Añadimos etiquetas al eje x y al eje y
plt.xlabel('Clase')
plt.ylabel('Cantidad')

# Añadimos un título a la gráfica
plt.title('Distribución de clases en el conjunto de datos de cáncer de seno')

# Mostramos la gráfica
plt.show()

#creamos la segunda grfica 
#segunda grfica 3D

fig = plt.figure(1, figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)

X_reduced = PCA(n_components=3).fit_transform(cancer.data)
ax.scatter(
    X_reduced[:, 0],
    X_reduced[:, 1],
    X_reduced[:, 2],
    c=cancer.target,
    s=40,
)

ax.set_title("Grafica 3d para el cancer")
ax.set_xlabel("1st Eigenvector")
ax.xaxis.set_ticklabels([])
ax.set_ylabel("2nd Eigenvector")
ax.yaxis.set_ticklabels([])
ax.set_zlabel("3rd Eigenvector")
ax.zaxis.set_ticklabels([])

plt.show()



#----------------------------------------------------------------iris
iris = datasets.load_iris()
#print(iris)
print("informacion del cojunto de entrada")
print(iris.keys())
#Verificamos las características del conjunto de datos de entrada 
print('Características del conjunto de datos de entrada:') 
print(iris.DESCR) 
print("Datos de Iris han finalizado..........................")




colors = ['red', 'blue', 'yellow']

# Creamos la gráfica de dispersión con los colores personalizados
_, ax = plt.subplots()
#iteramos los coress en el rango del arreglo
for i in range(len(iris.target_names)):
    indices = iris.target == i
    ax.scatter(iris.data[indices, 0], iris.data[indices, 1], c=colors[i], label=iris.target_names[i])

ax.set(xlabel=iris.feature_names[0], ylabel=iris.feature_names[1])
ax.legend()
plt.show()


#segunda grfica 3D

fig = plt.figure(1, figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)

X_reduced = PCA(n_components=3).fit_transform(iris.data)
ax.scatter(
    X_reduced[:, 0],
    X_reduced[:, 1],
    X_reduced[:, 2],
    c=iris.target,
    s=40,
)

ax.set_title("Iris 3D grafica ")
ax.set_xlabel("1st Eigenvector")
ax.xaxis.set_ticklabels([])
ax.set_ylabel("2nd Eigenvector")
ax.yaxis.set_ticklabels([])
ax.set_zlabel("3rd Eigenvector")
ax.zaxis.set_ticklabels([])

plt.show()

#...................................................Diabetes 

diabetes = datasets.load_diabetes()
print("informacion del cojunto de entrada")
print(diabetes.keys())
X, y = diabetes.data, diabetes.target
#Verificamos las características del conjunto de datos de entrada 
print('Características del conjunto de datos de entrada:') 
print(diabetes.DESCR) 
print("Datos de Diabetes han finalizado..........................")

#crear primera tabla 
ridge = RidgeCV(alphas=np.logspace(-6, 6, num=5)).fit(X, y)
importance = np.abs(ridge.coef_)
feature_names = np.array(diabetes.feature_names)
plt.bar(height=importance, x=feature_names)
plt.title("Coeficientes de importancia Diabetes")
plt.show()


#segunda grfica 

fig = plt.figure(1, figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)

X_reduced = PCA(n_components=3).fit_transform(diabetes.data)
ax.scatter(
    X_reduced[:, 0],
    X_reduced[:, 1],
    X_reduced[:, 2],
    c=diabetes.target,
    s=40,
)

ax.set_title("Grafica Diabetes 3D")
ax.set_xlabel("1st Eigenvector")
ax.xaxis.set_ticklabels([])
ax.set_ylabel("2nd Eigenvector")
ax.yaxis.set_ticklabels([])
ax.set_zlabel("3rd Eigenvector")
ax.zaxis.set_ticklabels([])

plt.show()

#..............................................Vinos

vinos = datasets.load_wine()
#print(iris)
print("informacion del cojunto de entrada")
print(vinos.keys())
#Verificamos las características del conjunto de datos de entrada 
print('Características del conjunto de datos de entrada:') 
print(vinos.DESCR) 
print("Datos de Iris han finalizado..........................")


colors = ['red', 'blue', 'yellow']

# Creamos la gráfica de dispersión con los colores personalizados
_, ax = plt.subplots()
#iteramos los coress en el rango del arreglo
for i in range(len(vinos.target_names)):
    indices = vinos.target == i
    ax.scatter(vinos.data[indices, 0], vinos.data[indices, 1], c=colors[i], label=vinos.target_names[i])

ax.set(xlabel=vinos.feature_names[0], ylabel=vinos.feature_names[1])
ax.legend()
plt.show()



#segunda grfica 

fig = plt.figure(1, figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)

X_reduced = PCA(n_components=3).fit_transform(vinos.data)
ax.scatter(
    X_reduced[:, 0],
    X_reduced[:, 1],
    X_reduced[:, 2],
    c=vinos.target,
    s=40,
)

ax.set_title("Grafica Vinos 3D")
ax.set_xlabel("1st Eigenvector")
ax.xaxis.set_ticklabels([])
ax.set_ylabel("2nd Eigenvector")
ax.yaxis.set_ticklabels([])
ax.set_zlabel("3rd Eigenvector")
ax.zaxis.set_ticklabels([])

plt.show()
