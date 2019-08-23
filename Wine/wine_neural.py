import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from Print_Timer_Results import *
from plot_learning_curve import drawLearningCurve
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Start timer
start_time = time.time()

# Load the data
from wine_data import X_train, X_test, y_train, y_test

# Scale the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Define the classifier
nn = MLPClassifier(solver='lbfgs', random_state=1)
grid_params = {'alpha': [0.05, 0.01, 0.005, 0.001]
              ,'hidden_layer_sizes': [3, 4, 7, 8, 12, 15, 20, 30, 50]
              }
clf = GridSearchCV(nn, param_grid=grid_params, cv=3)

# Run the classifier
clf.fit(X_train, y_train)

# Identify training accuracy
y_train_pred = clf.predict(X_train)
train_acc = np.sum(y_train == y_train_pred, axis=0) / X_train.shape[0]
print('Training accuracy: %.2f%%' % (train_acc * 100))

# Identify test set accuracy
y_test_pred = clf.predict(X_test)
test_acc = np.sum(y_test == y_test_pred, axis=0) / X_test.shape[0]
print('Test accuracy: %.2f%%' % (test_acc * 100))

# Draw learning curve
drawLearningCurve(clf, X_train, X_test, y_train, y_test, min_size=1000, numpoints=50)
plt.savefig('Neural Network Learning Curve.png', bbox_inches='tight')

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

# Show learning curve
scoreplot = plt.subplot()
for ind, i in enumerate(grid_params['hidden_layer_sizes']):
    # print('hidden_layer_sizes: ' + str(i))
    # print('alpha:' + str(grid_params['alpha']))
    # print('Score:' + str(scores[:,ind]))
    scoreplot.plot(grid_params['alpha'], scores[:,ind], label='hidden_layer_sizes: ' + str(i))
scoreplot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
scoreplot.set_xlabel('alpha')
scoreplot.set_xlim([0,0.05])
scoreplot.set_ylabel('Mean score')
scoreplot.set_title('Validation Curve')
plt.savefig('Neural Network Validation Curve.png', bbox_inches='tight')