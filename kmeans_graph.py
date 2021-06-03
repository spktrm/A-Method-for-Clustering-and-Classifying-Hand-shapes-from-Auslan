import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from alive_progress import alive_bar
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from rotate import normalise, rotate_3d

# pca params
n_PCA = 15
n_Kmeans = 10
d = '2d'

# file = "Basic words - Auslan"
# file = "Auslan COVID-19 Update - 29 Jan 2021"
# file = "WIN_20210418_10_11_51_Pro"
file = "WIN_20210422_17_16_38_Pro"

print("loading hand csv file...")
df = pd.read_csv(file + "/right_hand_keypoints.csv")
df = np.array(df)
df = df[:, 1:]
for i in range(len(df)):
    df[i][0::3] -=  df[i][0]
    df[i][1::3] -=  df[i][1]
    df[i][2::3] -=  df[i][2]

    pts = np.vstack([df[i][0::3], df[i][1::3], df[i][2::3]]).T
    pts = normalise(pts)

    df[i] = rotate_3d(pts).flatten()

print("hand csv file loaded!\n")

n_components = min(n_PCA, min(df.shape[0], df.shape[1]))
print("performing PCA on hands for first {} components...".format(n_components))
pcaf = PCA(n_components=n_components)
df = pcaf.fit_transform(df)
if d == '2d':
    var = round(np.sum(pcaf.explained_variance_ratio_[:2]) * 100, 2)
    print(("{}"+'%'+" of variance explained with first {} components...").format(var, 2))
else:
    var = round(np.sum(pcaf.explained_variance_ratio_[:3]) * 100, 2)
    print(("{}"+'%'+" of variance explained with first {} components...").format(var, 3))
print("PCA done.\n")

print("performing kmeans clustering on hand data...")
# Initialize the class object

df = pcaf.fit_transform(df)

kmeans = KMeans(n_clusters=n_Kmeans)
kmeans.fit(df)
# predict the labels of clusters.
labels = kmeans.fit_predict(df)

# Getting unique labels
u_labels = np.unique(labels)

# Getting the Centroids
centroids = kmeans.cluster_centers_
u_labels = np.unique(labels)

fig = plt.figure()
if d == '2d':
    ax1 = fig.add_subplot()
    for i in u_labels:
        ax1.scatter(df[labels == i, 0], df[labels == i, 1], label=i)
    ax1.scatter(centroids[:, 0], centroids[:, 1], s=80, color='k')
else:
    ax1 = fig.add_subplot(1, 1, 1, projection='3d')
    for i in u_labels:
        ax1.scatter(df[labels == i, 0], df[labels == i, 1], df[labels == i, 2], label=i)
    ax1.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], s=80, color='k')

ax1.legend()
plt.show()

print("kmeans clusters calculated!\n")
