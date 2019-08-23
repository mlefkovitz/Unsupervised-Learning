from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
from Print_Timer_Results import *
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.random_projection import GaussianRandomProjection as ProjectionAlgorithm
from mpl_toolkits.mplot3d import Axes3D
from Bayesian_Information_Criterion import *
import warnings
import time
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Start timer
start_time = time.time()

# Load the data
from wine_data import X, y, X_train,  X_test, y_train, y_test

# Scale the data
scaler = StandardScaler()
scaler.fit(X)
X_train_std = scaler.transform(X)
X_test_std = scaler.transform(X)
X_toCluster = X_train_std
y_inputs = y

# Reduce Dimensionality (Randomized Projection)
projection = ProjectionAlgorithm(n_components=9)
X_toCluster = projection.fit_transform(X_toCluster)

######
# Run em clustering with 2 clusters and plot
######
cluster = GaussianMixture(random_state=0, n_components=2).fit(X_toCluster)
cluster_labels = cluster.predict(X_toCluster)

X_transformed = np.dot(X_toCluster, np.transpose(cluster.means_))

# print diagnostics
print('X_toCluster.shape \n', X_toCluster.shape)
print('X_transformed.shape \n', X_transformed.shape)
print('Labels \n', cluster_labels)
print('Weights \n', cluster.weights_)
print('Means \n', cluster.means_)
print('Covariances \n', cluster.covariances_)
print('Log-likelihood of the best fit of EM \n', cluster.lower_bound_)

print('Prediction matrix \n', cluster.predict(X_toCluster))
print('Prediction matrix shape \n', cluster.predict(X_toCluster).shape)
print('Actual y labels \n', y_inputs)
print('Prediction = actuals \n', cluster.predict(X_toCluster) == y_inputs)
print('Prediction != actuals \n', cluster.predict(X_toCluster) != y_inputs)
print('Prediction = actuals \n', np.sum(cluster.predict(X_toCluster) == y_inputs))
print('Prediction != actuals \n', np.sum(cluster.predict(X_toCluster) != y_inputs))
print('Prediction = actuals % \n', np.sum(cluster.predict(X_toCluster) == y_inputs)/cluster.predict(X_toCluster).shape[0])
print('Prediction != actuals % \n', np.sum(cluster.predict(X_toCluster) != y_inputs)/cluster.predict(X_toCluster).shape[0])
print('Score \n', cluster.score(X_toCluster))


# Save plot with clusters as axes
# with plt.style.context('seaborn-whitegrid'):
#     plt.figure(figsize=(6, 4))
#     for yval, lab, col in zip((0, 1),
#                               ('<=50K', '>50K'),
#                               ('blue', 'red')):
#         plt.scatter(X_transformed[y_inputs==yval,0],
#                     X_transformed[y_inputs==yval,1],
#                     label=lab,
#                     c=col)
#     plt.xlabel('Cluster 1')
#     plt.ylabel('Cluster 2')
#     plt.legend(loc='best')
#     plt.title('EM clustering')
#     plt.tight_layout()
#     #plt.show()
#     plt.savefig('EM 2D clustering - RP Reduced.png', bbox_inches='tight')

# Save plot with PCA components as axes highlighting clusters
with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))
    for cluster, lab, col in zip((0, 1),
                              ('Cluster 1', 'Cluster 2'),
                              ('blue', 'red')):
        plt.scatter(X_toCluster[cluster_labels==cluster,0],
                    X_toCluster[cluster_labels==cluster,1],
                    label=lab,
                    c=col)
    plt.xlabel('Principle Component 1')
    plt.ylabel('Principle Component 2')
    #plt.xlim(-1000, 35000)
    plt.legend(loc='best')
    plt.title('2-means clustering - 2 principle components')
    plt.tight_layout()
    #plt.show()
    plt.savefig('EM 2D clustering - 2 components - RP Reduced.png', bbox_inches='tight')

Stop_Timer(start_time)

######
# Run em clustering with 13 clusters
######
cluster = GaussianMixture(random_state=0, n_components=13).fit(X_toCluster)
cluster_labels = cluster.predict(X_toCluster)

X_transformed = np.dot(X_toCluster, np.transpose(cluster.means_))

# print diagnostics
print('X_toCluster.shape \n', X_toCluster.shape)
print('X_transformed.shape \n', X_transformed.shape)
print('Labels \n', cluster_labels)
print('Weights \n', cluster.weights_)
print('Means \n', cluster.means_)
print('Covariances \n', cluster.covariances_)
print('Log-likelihood of the best fit of EM \n', cluster.lower_bound_)
print('Score \n', cluster.score(X_toCluster))

Stop_Timer(start_time)

######
# Run em clustering with 99 clusters
######
cluster = GaussianMixture(random_state=0, n_components=99).fit(X_toCluster)
cluster_labels = cluster.predict(X_toCluster)

X_transformed = np.dot(X_toCluster, np.transpose(cluster.means_))

# print diagnostics
print('X_toCluster.shape \n', X_toCluster.shape)
print('X_transformed.shape \n', X_transformed.shape)
print('Labels \n', cluster_labels)
print('Weights \n', cluster.weights_)
print('Means \n', cluster.means_)
print('Covariances \n', cluster.covariances_)
print('Log-likelihood of the best fit of EM \n', cluster.lower_bound_)
print('Score \n', cluster.score(X_toCluster))

Stop_Timer(start_time)