from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from Print_Timer_Results import *
from sklearn.decomposition import FastICA as ProjectionAlgorithm
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

# Reduce Dimensionality (ICA)
projection = ProjectionAlgorithm(n_components=29)
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
    plt.savefig('2-means clustering - ICA Reduced.png', bbox_inches='tight')

# Save plot with ICA components as axes highlighting clusters
with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))
    for cluster, lab, col in zip((0, 1),
                              ('Cluster 1', 'Cluster 2'),
                              ('blue', 'red')):
        plt.scatter(X_toCluster[kmeans.labels_==cluster,0],
                    X_toCluster[kmeans.labels_==cluster,1],
                    label=lab,
                    c=col)
    plt.xlabel('Independent Component 1')
    plt.ylabel('Independent Component 2')
    #plt.xlim(-1000, 35000)
    plt.legend(loc='best')
    plt.title('2-means clustering - 2 independent components')
    plt.tight_layout()
    #plt.show()
    plt.savefig('2-means clustering - 2 components - clusters - ICA Reduced.png', bbox_inches='tight')

Stop_Timer(start_time)

######
# Run k-means clustering with 3 clusters and plot
######
kmeans = KMeans(n_clusters=3, random_state=0).fit(X_toCluster)
X_transformed = KMeans(n_clusters=3, random_state=0).fit_transform(X_toCluster)

# print diagnostics
print('X_toCluster.shape \n', X_toCluster.shape)
print('X_transformed.shape \n', X_transformed.shape)
print('Labels \n', kmeans.labels_)
print('Cluster Centers \n', kmeans.cluster_centers_)
print('kmeans output \n', kmeans)
print('Score \n', kmeans.score(X_toCluster))

# Save plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for yval, lab, col in zip((0, 1),
                       ('<=50K', '>50K'),
                    ('blue', 'red')):
    ax.scatter(X_transformed[y_inputs==yval,0],
                X_transformed[y_inputs==yval,1],
                X_transformed[y_inputs == yval, 2],
                label=lab,
                c=col)
ax.set_xlabel('Cluster 1')
ax.set_ylabel('Cluster 2')
ax.set_zlabel('Cluster 3')
ax.legend(loc='best')
ax.set_title('3-means clustering - ICA Reduced')
#plt.show()
plt.savefig('3-means clustering  - ICA Reduced.png', bbox_inches='tight')

# Save plot with ICA components as axes highlighting clusters
with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))
    for cluster, lab, col in zip((0, 1, 2),
                              ('Cluster 1', 'Cluster 2','Cluster 3'),
                              ('blue', 'red', 'green')):
        plt.scatter(X_toCluster[kmeans.labels_==cluster,0],
                    X_toCluster[kmeans.labels_==cluster,1],
                    label=lab,
                    c=col)
    plt.xlabel('Independent Component 1')
    plt.ylabel('Independent Component 2')
    #plt.xlim(-1000, 35000)
    plt.legend(loc='best')
    plt.title('3-means clustering - 2 independent components')
    plt.tight_layout()
    #plt.show()
    plt.savefig('3-means clustering - 2 components - clusters - ICA Reduced.png', bbox_inches='tight')

Stop_Timer(start_time)

######
# Run k-means clustering with 15 clusters
######
kmeans = KMeans(n_clusters=15, random_state=0).fit(X_toCluster)
X_transformed = KMeans(n_clusters=15, random_state=0).fit_transform(X_toCluster)

# print diagnostics
print('X_toCluster.shape \n', X_toCluster.shape)
print('X_transformed.shape \n', X_transformed.shape)
print('Labels \n', kmeans.labels_)
print('Cluster Centers \n', kmeans.cluster_centers_)
print('kmeans output \n', kmeans)
print('Score \n', kmeans.score(X_toCluster))

# Save plot with ICA components as axes highlighting clusters
with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))
    for cluster, lab, col in zip((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14),
                              ('Cluster 1', 'Cluster 2','Cluster 3','Cluster 4','Cluster 5','Cluster 6','Cluster 7',
                               'Cluster 8','Cluster 9','Cluster 10','Cluster 11','Cluster 12','Cluster 13','Cluster 14',
                               'Cluster 15'),
                              ('blue', 'red', 'green', 'yellow', 'pink', 'purple', 'black', 'white', 'grey', 'brown',
                               'magenta', 'cyan', 'blue', 'red', 'green')):
        plt.scatter(X_toCluster[kmeans.labels_==cluster,0],
                    X_toCluster[kmeans.labels_==cluster,1],
                    label=lab,
                    c=col)
    plt.xlabel('Independent Component 1')
    plt.ylabel('Independent Component 2')
    #plt.xlim(-1000, 35000)
    plt.legend(loc='best')
    plt.title('15-means clustering - 2 independent components')
    plt.tight_layout()
    #plt.show()
    plt.savefig('15-means clustering - 2 components - clusters - ICA Reduced.png', bbox_inches='tight')

Stop_Timer(start_time)
