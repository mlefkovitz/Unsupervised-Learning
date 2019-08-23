import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import GaussianRandomProjection as ProjectionAlgorithm
from sklearn.model_selection import GridSearchCV
from sklearn.mixture import GaussianMixture
from Print_Timer_Results import *
from plot_learning_curve import drawLearningCurve
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Start timer
start_time = time.time()

# Load the data
from income_data import X_train, X_test, y_train, y_test

# Scale the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_toTransform = X_train

# Reduce Dimensionality
projection = ProjectionAlgorithm(n_components=22)
X_transformed = projection.fit_transform(X_train)
X_testTransformed = projection.transform(X_test)

# Run em clustering with 2 clusters and plot
cluster = GaussianMixture(random_state=0, n_components=99).fit(X_transformed)
cluster_labels = cluster.predict(X_transformed)
X_transformed = np.dot(X_transformed, np.transpose(cluster.means_))
X_testTransformed = np.dot(X_testTransformed, np.transpose(cluster.means_))

# Define the classifier
nn = MLPClassifier(solver='lbfgs', random_state=1, alpha=0.005, hidden_layer_sizes=3)
grid_params = {'alpha': [0.005]
              ,'hidden_layer_sizes': [3]
              }
clf = GridSearchCV(nn, param_grid=grid_params,cv=3)

# Run the classifier
clf.fit(X_transformed, y_train)

# Identify training accuracy
y_train_pred = clf.predict(X_transformed)
train_acc = np.sum(y_train == y_train_pred, axis=0) / X_train.shape[0]
print('Training accuracy: %.2f%%' % (train_acc * 100))

# Identify test set accuracy
y_test_pred = clf.predict(X_testTransformed)
test_acc = np.sum(y_test == y_test_pred, axis=0) / X_test.shape[0]
print('Test accuracy: %.2f%%' % (test_acc * 100))

# Draw learning curve
drawLearningCurve(clf, X_transformed, X_testTransformed, y_train, y_test, min_size=100, numpoints=50)
plt.savefig('Neural Network Learning Curve - RP EM-99.png', bbox_inches='tight')

# Print diagnostics
print(clf.best_score_)
print(clf.best_params_)
print(clf.best_estimator_)
print('gridscores:')
print(clf.grid_scores_)

# Print diagnostics
scores = [x[1] for x in clf.grid_scores_]
print('scores:')
print(scores)
scores = np.array(scores).reshape(len(grid_params['alpha']), len(grid_params['hidden_layer_sizes']))
print('scores:')
print(scores)

Stop_Timer(start_time)

# # Show learning curve
# scoreplot = plt.subplot()
# for ind, i in enumerate(grid_params['hidden_layer_sizes']):
#     # print('hidden_layer_sizes: ' + str(i))
#     # print('alpha:' + str(grid_params['alpha']))
#     # print('Score:' + str(scores[:,ind]))
#     scoreplot.plot(grid_params['alpha'], scores[:,ind], label='hidden_layer_sizes: ' + str(i))
# scoreplot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# scoreplot.set_xlabel('alpha')
# scoreplot.set_xlim([0,0.05])
# scoreplot.set_ylabel('Mean score')
# scoreplot.set_title('Validation Curve')
# plt.savefig('Neural Network Validation Curve.png', bbox_inches='tight')