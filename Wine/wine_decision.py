from sklearn.metrics import accuracy_score
from PrunedTrees import dtclf_pruned
from sklearn.model_selection import GridSearchCV
from Print_Timer_Results import *
from plot_learning_curve import drawLearningCurve
import matplotlib.pyplot as plt
import warnings
import time
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Start timer
start_time = time.time()

# Load the data
from wine_data import X_train,  X_test, y_train, y_test

# Define the classifier
tree = dtclf_pruned()
parameters = {'alpha': [-1, -0.6, -0.3, -0.1, -0.06, -0.03, -0.01, -0.006, -0.003, -0.001, 0, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1]
             }
clf = GridSearchCV(tree, parameters)

# Run the classifier
clf.fit(X_train, y_train)

# Identify training and test accuracy
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)
print('Decision Tree train/test accuracy: %.3f/%.3f' % (train_accuracy, test_accuracy))

# Export to .dot file
from sklearn.tree import export_graphviz
export_graphviz(clf.best_estimator_, out_file = 'Wine2pruned.dot', feature_names=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality'])

# Draw learning curve
drawLearningCurve(clf, X_train, X_test, y_train, y_test, min_size=2000, numpoints=100)
plt.savefig('Decision Tree Learning Curve.png', bbox_inches='tight')

# Print diagnostics
print(clf.best_score_)
print(clf.best_params_)
print(clf.best_estimator_)
print('gridscores:')
print(clf.grid_scores_)

Stop_Timer(start_time)

# Show learning curve
scores = [x[1] for x in clf.grid_scores_]
scoreplot = plt.subplot()
scoreplot.plot(parameters['alpha'], scores)
scoreplot.set_xlabel('alpha')
scoreplot.set_ylabel('Mean score')
scoreplot.set_title('Validation Curve')
plt.savefig('Decision Tree Validation Curve.png', bbox_inches='tight')
