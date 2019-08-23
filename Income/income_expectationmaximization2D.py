from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
from Print_Timer_Results import *
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
from mpl_toolkits.mplot3d import Axes3D
from Bayesian_Information_Criterion import *
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

print('Prediction matrix \n', cluster.predict(X_test_std))
print('Prediction matrix shape \n', cluster.predict(X_test_std).shape)
print('Actual y labels \n', y_inputs)
print('Prediction = actuals \n', cluster.predict(X_test_std) == y_inputs)
print('Prediction != actuals \n', cluster.predict(X_test_std) != y_inputs)
print('Prediction = actuals \n', np.sum(cluster.predict(X_test_std) == y_inputs))
print('Prediction != actuals \n', np.sum(cluster.predict(X_test_std) != y_inputs))
print('Prediction = actuals % \n', np.sum(cluster.predict(X_test_std) == y_inputs)/cluster.predict(X_test_std).shape[0])
print('Prediction != actuals % \n', np.sum(cluster.predict(X_test_std) != y_inputs)/cluster.predict(X_test_std).shape[0])
print('Score \n', cluster.score(X_toCluster))


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
    plt.title('EM clustering')
    plt.tight_layout()
    #plt.show()
    plt.savefig('EM 2D clustering.png', bbox_inches='tight')

# Save plot with married (15) and capital gains (3) as axes highlighting clusters
with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))
    for clust, lab, col in zip((0, 1),
                              ('Cluster 1', 'Cluster 2'),
                              ('blue', 'red')):
        plt.scatter(X[cluster_labels==clust,3],
                    X[cluster_labels==clust,15],
                    label=lab,
                    c=col)
    plt.xlabel('Capital Gains')
    plt.ylabel('Married?')
    plt.xlim(-1000, 35000)
    plt.legend(loc='best')
    plt.title('EM clustering - best 2 factors')
    plt.tight_layout()
    #plt.show()
    plt.savefig('EM clustering - best 2 features - clusters.png', bbox_inches='tight')

# Save plot with married (15) and capital gains (3) as axes highlighting y values
with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))
    for yval, lab, col in zip((0, 1),
                              ('<=50K', '>50K'),
                              ('blue', 'red')):
        plt.scatter(X[y_inputs==yval,3],
                    X[y_inputs==yval,15],
                    label=lab,
                    c=col)
    plt.xlabel('Capital Gains')
    plt.ylabel('Married?')
    plt.xlim(-1000, 35000)
    plt.legend(loc='best')
    plt.title('EM clustering - best 2 features')
    plt.tight_layout()
    #plt.show()
    plt.savefig('EM clustering - best 2 features - y values.png', bbox_inches='tight')

# Save plot with married (15) and capital gains (3) as axes highlighting everything
with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))
    for yval_cluster, lab, col in zip(([0, 0], [1, 0], [0, 1], [1, 1]),
                              ('<=50K cluster 1', '>50K cluster 1', '<=50K cluster 2', '>50K cluster 2'),
                              ('blue', 'green', 'red', 'pink')):
        plt.scatter(X[(y_inputs==yval_cluster[0]) & (cluster_labels==yval_cluster[1]),3],
                    X[(y_inputs==yval_cluster[0]) & (cluster_labels==yval_cluster[1]),15],
                    label=lab,
                    c=col)
    plt.xlabel('Capital Gains')
    plt.ylabel('Married?')
    plt.xlim(-1000, 35000)
    plt.legend(loc='best')
    plt.title('EM clustering - best 2 features')
    plt.tight_layout()
    #plt.show()
    plt.savefig('EM clustering - best 2 features - y values and clusters.png', bbox_inches='tight')

# Save plot with married (15), capital gains (3), and eduation (1) as axes highlighting clusters
# Save plot
fig = plt.figure()
ax = Axes3D(fig)
for clust, lab, col in zip((0, 1),
                              ('Cluster 1', 'Cluster 2'),
                              ('blue', 'red')):
    ax.scatter(X[cluster_labels==clust,3],
                X[cluster_labels==clust,15],
                X[cluster_labels == clust, 1],
                label=lab,
                c=col)
ax.set_xlabel('Capital Gains')
ax.set_ylabel('Married?')
ax.set_zlabel('Education')
ax.legend(loc='best')
ax.set_title('EM lustering')
#plt.show()
plt.savefig('EM clustering - best 3 features - capital gains, married, and education.png', bbox_inches='tight')

Stop_Timer(start_time)
