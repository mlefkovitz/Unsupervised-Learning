from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from Print_Timer_Results import *
from sklearn.decomposition import TruncatedSVD as ProjectionAlgorithm
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

# Reduce Dimensionality (Truncated SVD)
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
    plt.savefig('2-means clustering - Truncated SVD Reduced.png', bbox_inches='tight')

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
    plt.savefig('2-means clustering - 2 components - clusters - Truncated SVD Reduced.png', bbox_inches='tight')

# # Save plot with married (15) and capital gains (3) as axes highlighting clusters
# with plt.style.context('seaborn-whitegrid'):
#     plt.figure(figsize=(6, 4))
#     for cluster, lab, col in zip((0, 1),
#                               ('Cluster 1', 'Cluster 2'),
#                               ('blue', 'red')):
#         plt.scatter(X[kmeans.labels_==cluster,3],
#                     X[kmeans.labels_==cluster,15],
#                     label=lab,
#                     c=col)
#     plt.xlabel('Capital Gains')
#     plt.ylabel('Married?')
#     plt.xlim(-1000, 35000)
#     plt.legend(loc='best')
#     plt.title('2-means clustering - best 2 factors')
#     plt.tight_layout()
#     #plt.show()
#     plt.savefig('2-means clustering - best 2 factors - clusters.png', bbox_inches='tight')
#
# # Save plot with married (15) and capital gains (3) as axes highlighting y values
# with plt.style.context('seaborn-whitegrid'):
#     plt.figure(figsize=(6, 4))
#     for yval, lab, col in zip((0, 1),
#                               ('<=50K', '>50K'),
#                               ('blue', 'red')):
#         plt.scatter(X[y_inputs==yval,3],
#                     X[y_inputs==yval,15],
#                     label=lab,
#                     c=col)
#     plt.xlabel('Capital Gains')
#     plt.ylabel('Married?')
#     plt.xlim(-1000, 35000)
#     plt.legend(loc='best')
#     plt.title('2-means clustering - best 2 factors')
#     plt.tight_layout()
#     #plt.show()
#     plt.savefig('2-means clustering - best 2 factors - y values.png', bbox_inches='tight')
#
# # Save plot with married (15) and capital gains (3) as axes highlighting everything
# with plt.style.context('seaborn-whitegrid'):
#     plt.figure(figsize=(6, 4))
#     for yval_cluster, lab, col in zip(([0, 0], [1, 0], [0, 1], [1, 1]),
#                               ('<=50K cluster 1', '>50K cluster 1', '<=50K cluster 2', '>50K cluster 2'),
#                               ('blue', 'green', 'red', 'pink')):
#         plt.scatter(X[(y_inputs==yval_cluster[0]) & (kmeans.labels_==yval_cluster[1]),3],
#                     X[(y_inputs==yval_cluster[0]) & (kmeans.labels_==yval_cluster[1]),15],
#                     label=lab,
#                     c=col)
#     plt.xlabel('Capital Gains')
#     plt.ylabel('Married?')
#     plt.xlim(-1000, 35000)
#     plt.legend(loc='best')
#     plt.title('2-means clustering - best 2 factors')
#     plt.tight_layout()
#     #plt.show()
#     plt.savefig('2-means clustering - best 2 factors - y values and clusters.png', bbox_inches='tight')
#
# # Save plot with married (15), capital gains (3), and eduation (1) as axes highlighting clusters
# # Save plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for cluster, lab, col in zip((0, 1),
#                               ('Cluster 1', 'Cluster 2'),
#                               ('blue', 'red')):
#     ax.scatter(X[kmeans.labels_==cluster,3],
#                 X[kmeans.labels_==cluster,15],
#                 X[kmeans.labels_ == cluster, 1],
#                 label=lab,
#                 c=col)
# ax.set_xlabel('Capital Gains')
# ax.set_ylabel('Married?')
# ax.set_zlabel('Education')
# ax.legend(loc='best')
# ax.set_title('2-means clustering')
# #plt.show()
# plt.savefig('2-means clustering by capital gains, married, and education.png', bbox_inches='tight')

# ######
# # Run k-means clustering with 3 clusters and plot
# ######
# kmeans = KMeans(n_clusters=3, random_state=0).fit(X_toCluster)
# X_transformed = KMeans(n_clusters=3, random_state=0).fit_transform(X_toCluster)
#
# # print diagnostics
# print('X_toCluster.shape \n', X_toCluster.shape)
# print('X_transformed.shape \n', X_transformed.shape)
# print('Labels \n', kmeans.labels_)
# print('Prediction matrix \n', kmeans.predict(X_test_std))
# print('Prediction matrix shape \n', kmeans.predict(X_test_std).shape)
# print('Cluster Centers \n', kmeans.cluster_centers_)
# print('kmeans output \n', kmeans)
# print('Score \n', kmeans.score(X_toCluster))
#
# # Save plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for yval, lab, col in zip((0, 1),
#                        ('<=50K', '>50K'),
#                     ('blue', 'red')):
#     ax.scatter(X_transformed[y_train==yval,0],
#                 X_transformed[y_train==yval,1],
#                 X_transformed[y_train == yval, 2],
#                 label=lab,
#                 c=col)
# ax.set_xlabel('Cluster 1')
# ax.set_ylabel('Cluster 2')
# ax.set_zlabel('Cluster 3')
# ax.legend(loc='best')
# ax.set_title('3-means clustering')
# #plt.show()
# plt.savefig('3-means clustering.png', bbox_inches='tight')


Stop_Timer(start_time)
