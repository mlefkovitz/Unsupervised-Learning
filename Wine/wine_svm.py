from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
from plot_learning_curve import drawLearningCurve
from Print_Timer_Results import *
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
svm = SVC(random_state=1)
parameters = {'kernel':('linear', 'rbf')
             ,'C':[1, 10]
             ,'gamma':(0.1, 0.5)
             }
clf = GridSearchCV(svm, parameters)

# Run the classifier
clf.fit(X_train, y_train)

# Identify training and test accuracy
y_pred = clf.predict(X_test)
print('Misclassified samples: %d' % (y_test != y_pred).sum())
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)
print('SVM Kernel train/test accuracy: %.3f/%.3f' % (train_accuracy, test_accuracy))

# Draw learning curve
drawLearningCurve(clf, X_train, X_test, y_train, y_test, min_size=1000, numpoints=50)
plt.savefig('SVM Learning Curve.png', bbox_inches='tight')

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
scores = np.array(scores).reshape(len(parameters['C']),len(parameters['kernel']) * len(parameters['gamma']))
scores = scores.transpose()
print('scores:')
print(scores)

Stop_Timer(start_time)

# Show learning curve
test = [x[0] for x in clf.grid_scores_]
total = len(parameters['gamma'])*len(parameters['kernel'])
gammas = [x['gamma'] for x in test[0:total]]
kernels = [x['kernel'] for x in test[0:total]]
iterator = np.column_stack((gammas, kernels))
scoreplot = plt.subplot()
for ind, i in enumerate(iterator):
    # print('kernel: ' + str(i[1]) + '; gamma: ' + str(i[0]))
    # print('C:' + str(parameters['C']))
    # print('Score:' + str(scores[ind]))
    scoreplot.plot(parameters['C'], scores[ind], label='kernel: ' + str(i[1]) + '; gamma: ' + str(i[0]))
scoreplot.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=2)
scoreplot.set_xlabel('C')
scoreplot.set_ylabel('Mean score')
scoreplot.set_title('Validation Curve')
plt.savefig('SVM Validation Curve.png', bbox_inches='tight')