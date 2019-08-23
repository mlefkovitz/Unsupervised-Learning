from sklearn.metrics import accuracy_score
from PrunedTrees import dtclf_pruned
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from Print_Timer_Results import *
from plot_learning_curve import drawLearningCurve
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import time

# Start timer
start_time = time.time()

# Load the data
from wine_data import X_train, X_test, y_train, y_test

# Define the classifier
tree = dtclf_pruned(alpha=0.006)
ada = AdaBoostClassifier(base_estimator=tree, random_state=0)

parameters = {'n_estimators': [10, 20, 40, 70, 100, 150, 200]
             ,'learning_rate':[3, 1, 0.3, 0.1, 0.05, 0.01, 0.005, 0.001]
             }

clf = GridSearchCV(ada, parameters)

# Run the classifier
clf.fit(X_train, y_train)

# Identify training and test accuracy
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)
ada_train = accuracy_score(y_train, y_train_pred)
ada_test = accuracy_score(y_test, y_test_pred)
print('Ada boost train/test accuracies %.3f/%.3f' % (ada_train, ada_test))

# Draw learning curve
drawLearningCurve(clf, X_train, X_test, y_train, y_test, min_size=2000, numpoints=10)
plt.savefig('Boosting Learning Curve.png', bbox_inches='tight')

# Print diagnostics
print(clf.best_score_)
print(clf.best_params_)
print(clf.best_estimator_)
print('gridscores:')
print(clf.grid_scores_)
scores = [x[1] for x in clf.grid_scores_]
print('scores:')
print(scores)
scores = np.array(scores).reshape(len(parameters['learning_rate']), len(parameters['n_estimators']))
print('scores:')
print(scores)

Stop_Timer(start_time)

# Show learning curve
scoreplot = plt.subplot()
for ind, i in enumerate(parameters['learning_rate']):
    # print('learning_rate: ' + str(i))
    # print('n_estimators:' + str(parameters['n_estimators']))
    # print('Score:' + str(scores[ind]))
    scoreplot.plot(parameters['n_estimators'], scores[ind], label='learning_rate: ' + str(i))
scoreplot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
scoreplot.set_xlabel('n_estimators')
scoreplot.set_ylabel('Mean score')
scoreplot.set_title('Validation Curve')
plt.savefig('Boosting Validation Curve.png', bbox_inches='tight')

# from sklearn.tree import export_graphviz
# export_graphviz(clf.best_estimator_, out_file = 'Wine2boosted.dot', feature_names=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality'])