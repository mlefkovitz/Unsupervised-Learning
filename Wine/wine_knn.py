from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from Print_Timer_Results import *
import matplotlib.pyplot as plt
from plot_learning_curve import drawLearningCurve
import warnings
import time
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Start timer
start_time = time.time()

# Load the data
from wine_data import X_train, X_test, y_train, y_test

# Scale the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)

# Define the classifier
knn = KNeighborsClassifier()
parameters = {'n_neighbors': range(1,50)
             }
clf = GridSearchCV(knn, param_grid=parameters, cv=5)

# Run the classifier
clf.fit(X_train_std, y_train)

# Identify training and test accuracy
y_pred_train = clf.predict(X_train_std)
y_pred_test = clf.predict(X_test_std)
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)
print('KNN: train/test accuracy: %.3f/%.3f' % (train_accuracy, test_accuracy))

# Draw learning curve
drawLearningCurve(clf, X_train, X_test, y_train, y_test, min_size=1000, numpoints=50)
plt.savefig('KNN Learning Curve.png', bbox_inches='tight')

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
scoreplot.plot(parameters['n_neighbors'], scores)
scoreplot.set_xlabel('neighbors')
scoreplot.set_ylabel('Mean score')
scoreplot.set_title('Validation Curve')
plt.savefig('KNN Validation Curve.png', bbox_inches='tight')