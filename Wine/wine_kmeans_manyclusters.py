from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from Print_Timer_Results import *
from sklearn.metrics import silhouette_samples, silhouette_score
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


######
# Run k-means clustering with 1:n clusters determine scores for each
######
scores = []
silhouette_avg = []
BIC = []
maxClusters = 100
minClusters = 1
for i in range(minClusters,maxClusters):
    kmeans = KMeans(n_clusters=i+1, random_state=0)
    cluster_labels = kmeans.fit_predict(X_toCluster)
    scores.append(kmeans.score(X_toCluster))
    silhouette_avg.append(silhouette_score(X, cluster_labels))
    BIC.append(compute_bic(kmeans, X_toCluster))
    print('Number of clusters = ', i+1, 'Score =', kmeans.score(X_toCluster))
    print("For n_clusters =", i+1, "The average silhouette_score is :", silhouette_avg[i-minClusters])
    print("For n_clusters =", i+1, "The average bayesian information criterion is :", BIC[i-minClusters])


# print('Scores: \n', scores)
print('Max score: %.2f' % np.max(scores), 'with ', np.argmax(scores==np.max(scores))+2,'clusters')
print('Max silhouette score: %.2f' % np.max(silhouette_avg), 'with ', np.argmax(silhouette_avg==np.max(silhouette_avg))+2,'clusters')
print('Min bayesian information criterion: %.2f' % np.min(BIC), 'with ', np.argmin(BIC==np.min(BIC))+1,'clusters')

# Save score plot
with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))
    plt.plot(range(minClusters+1,maxClusters+1),
                scores,
             'bo',
             range(minClusters+1, maxClusters+1),
             scores,
             'k')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Score')
    plt.title('k-means Clustering Scores')
    plt.tight_layout()
    #plt.show()
    plt.savefig('k-means Clustering Scores.png', bbox_inches='tight')

# Save silhouette score plot
with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))
    plt.plot(range(minClusters+1,maxClusters+1),
             silhouette_avg,
             'bo',
             range(minClusters+1, maxClusters+1),
             silhouette_avg,
             'k')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('k-means Clustering Silhouette Scores')
    plt.tight_layout()
    #plt.show()
    plt.savefig('k-means Clustering Silhouette Scores.png', bbox_inches='tight')

# Save BIC plot
with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))
    plt.plot(range(minClusters+1,maxClusters+1),
             BIC,
             'bo',
             range(minClusters+1, maxClusters+1),
             BIC,
             'k')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Bayesian Information Criterion')
    plt.title('k-means Clustering BIC')
    plt.tight_layout()
    #plt.show()
    plt.savefig('k-means Clustering BIC.png', bbox_inches='tight')

Stop_Timer(start_time)


