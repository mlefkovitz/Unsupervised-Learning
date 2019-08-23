from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
from Print_Timer_Results import *
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.random_projection import GaussianRandomProjection as ProjectionAlgorithm
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

# Reduce Dimensionality (Randomized Projections)
projection = ProjectionAlgorithm(n_components=9)
X_toCluster = projection.fit_transform(X_toCluster)

######
# Run em clustering with 1:n clusters determine scores for each
######
scores = []
silhouette_avg = []
BIC = []
AIC = []
LL = []
maxClusters = 100
minClusters = 1
for i in range(minClusters,maxClusters):
    cluster = GaussianMixture(random_state=0, n_components=i+1).fit(X_toCluster)
    cluster_labels = cluster.predict(X_toCluster)
    scores.append(cluster.score(X_toCluster))
    LL.append(cluster.lower_bound_)
    silhouette_avg.append(silhouette_score(X, cluster_labels))
    BIC.append(cluster.bic(X_toCluster))
    AIC.append(cluster.aic(X_toCluster))
    print('Number of clusters = ', i+1, 'Score =', cluster.score(X_toCluster))
    print("For n_clusters =", i + 1, "The average Log-likelihood best fit is :", LL[i - minClusters])
    print("For n_clusters =", i + 1, "The average silhouette_score is :", silhouette_avg[i - minClusters])
    print("For n_clusters =", i+1, "The average bayesian information criterion is :", BIC[i-minClusters])
    print("For n_clusters =", i+1, "The average akaike information criterion is :", AIC[i-minClusters])


# print('Scores: \n', scores)
print('Max score: %.2f' % np.max(scores), 'with ', np.argmax(scores==np.max(scores))+2,'clusters')
print('Max Log-Likelihood: %.2f' % np.max(LL), 'with ', np.argmax(LL==np.max(LL))+2,'clusters')
print('Max silhouette score: %.2f' % np.max(silhouette_avg), 'with ', np.argmax(silhouette_avg==np.max(silhouette_avg))+2,'clusters')
print('Min bayesian information criterion: %.2f' % np.min(BIC), 'with ', np.argmin(BIC==np.min(BIC))+1,'clusters')
print('Min akaike information criterion: %.2f' % np.min(AIC), 'with ', np.argmin(AIC==np.min(AIC))+1,'clusters')

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
    plt.title('EM Clustering Scores')
    plt.tight_layout()
    #plt.show()
    plt.savefig('EM Clustering Scores - RP.png', bbox_inches='tight')

# Save LL plot
with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))
    plt.plot(range(minClusters+1,maxClusters+1),
             LL,
             'bo',
             range(minClusters+1, maxClusters+1),
             LL,
             'k')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Log-Likelihood')
    plt.title('EM Clustering Log-Likelihood')
    plt.tight_layout()
    #plt.show()
    plt.savefig('EM Clustering LL - RP.png', bbox_inches='tight')

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
    plt.title('EM Clustering Silhouette Scores')
    plt.tight_layout()
    #plt.show()
    plt.savefig('EM Clustering Silhouette Scores - RP.png', bbox_inches='tight')

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
    plt.title('EM Clustering BIC')
    plt.tight_layout()
    #plt.show()
    plt.savefig('EM Clustering BIC - RP.png', bbox_inches='tight')

# Save AIC plot
with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))
    plt.plot(range(minClusters+1,maxClusters+1),
             AIC,
             'bo',
             range(minClusters+1, maxClusters+1),
             AIC,
             'k')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Akaike Information Criterion')
    plt.title('EM Clustering AIC')
    plt.tight_layout()
    #plt.show()
    plt.savefig('EM Clustering AIC - RP.png', bbox_inches='tight')

Stop_Timer(start_time)

