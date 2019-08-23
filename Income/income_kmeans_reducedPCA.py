from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from Print_Timer_Results import *
from sklearn.decomposition import PCA as ProjectionAlgorithm
from mpl_toolkits.mplot3d import Axes3D
import warnings
import time
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Start timer
start_time = time.time()

# Load the data
from income_data import X, y, X_train,  X_test, y_train, y_test

# Scale the data
scaler = StandardScaler()
scaler.fit(X)
X_train_std = scaler.transform(X)
X_test_std = scaler.transform(X)
X_toCluster = X_train_std
y_inputs = y

# Reduce Dimensionality (PCA)
projection = ProjectionAlgorithm(n_components=34)
X_toCluster = projection.fit_transform(X_toCluster)

######
# Run k-means clustering with 2 clusters and plot
######
kmeans = KMeans(n_clusters=2, random_state=0).fit(X_toCluster)
X_transformed = KMeans(n_clusters=2, random_state=0).fit_transform(X_toCluster)

# print diagnostics
print('X_toCluster.shape \n', X_toCluster.shape)
print('X_transformed.shape \n', X_transformed.shape)
print('Labels \n', kmeans.labels_)
print('Prediction matrix \n', kmeans.predict(X_toCluster))
print('Prediction matrix shape \n', kmeans.predict(X_toCluster).shape)
print('Actual y labels \n', y_inputs)
print('Prediction = actuals \n', kmeans.predict(X_toCluster) == y_inputs)
print('Prediction != actuals \n', kmeans.predict(X_toCluster) != y_inputs)
print('Prediction = actuals \n', np.sum(kmeans.predict(X_toCluster) == y_inputs))
print('Prediction != actuals \n', np.sum(kmeans.predict(X_toCluster) != y_inputs))
print('Prediction = actuals % \n', np.sum(kmeans.predict(X_toCluster) == y_inputs)/kmeans.predict(X_toCluster).shape[0])
print('Prediction != actuals % \n', np.sum(kmeans.predict(X_toCluster) != y_inputs)/kmeans.predict(X_toCluster).shape[0])
print('Cluster Centers \n', kmeans.cluster_centers_)
print('kmeans output \n', kmeans)
print('Score \n', kmeans.score(X_toCluster))

# Save plot with clusters as axes
with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))
    for yval, lab, col in zip((0, 1),
                              ('<=50K', '>50K'),
                              ('blue', 'red')):
        plt.scatter(X_transformed[y_inputs==yval,0],
                    X_transformed[y_inputs==yval,1],
                    label=lab,
                    c=col)
    plt.xlabel('Cluster 1')
    plt.ylabel('Cluster 2')
    plt.legend(loc='best')
    plt.title('2-means clustering')
    plt.tight_layout()
    #plt.show()
    plt.savefig('2-means clustering - PCA Reduced.png', bbox_inches='tight')

# Save plot with PCA components as axes highlighting clusters
with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))
    for cluster, lab, col in zip((0, 1),
                              ('Cluster 1', 'Cluster 2'),
                              ('blue', 'red')):
        plt.scatter(X_toCluster[kmeans.labels_==cluster,0],
                    X_toCluster[kmeans.labels_==cluster,1],
                    label=lab,
                    c=col)
    plt.xlabel('Principle Component 1')
    plt.ylabel('Principle Component 2')
    #plt.xlim(-1000, 35000)
    plt.legend(loc='best')
    plt.title('2-means clustering - 2 principle components')
    plt.tight_layout()
    #plt.show()
    plt.savefig('2-means clustering - 2 components - clusters - PCA Reduced.png', bbox_inches='tight')


Stop_Timer(start_time)
