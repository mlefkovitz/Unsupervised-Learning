from sklearn.decomposition import TruncatedSVD as ProjectionAlgorithm
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from Print_Timer_Results import *
from ReconstructionError import *
import warnings
import time
from mpl_toolkits.mplot3d import Axes3D
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Start timer
start_time = time.time()

# Load the data
from income_data import X, y, X_train,  X_test, y_train, y_test

######
# Run Grid Search to find optimal components
######
# import packages
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from plot_learning_curve import drawLearningCurve

# Scale the data
scaler = StandardScaler()
scaler.fit(X)
X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)
X_toTransform = X_train_std
y_train = y_train
y_test = y_test

# Define the classifier
svm = SVC(random_state=1, kernel='linear', gamma=0.1, C=10)
pipe = Pipeline([
    ('reduce_dim', ProjectionAlgorithm())
     ,('classify', svm)
])
N_FEATURES_OPTIONS = range(2,46)
parameters = {  'reduce_dim__n_components':N_FEATURES_OPTIONS,
             }
clf = GridSearchCV(pipe, cv=3, param_grid=parameters)

# Run the classifier
clf.fit(X_train_std, y_train)

# Identify training and test accuracy
y_pred = clf.predict(X_test_std)
print('Misclassified samples: %d' % (y_test != y_pred).sum())
y_pred_train = clf.predict(X_train_std)
y_pred_test = clf.predict(X_test_std)
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)
print('SVM Kernel train/test accuracy: %.3f/%.3f' % (train_accuracy, test_accuracy))

mean_scores = np.array(clf.cv_results_['mean_test_score'])
# scores are in the order of param_grid iteration, which is alphabetical
# mean_scores = mean_scores.reshape(len(C_OPTIONS), -1, len(N_FEATURES_OPTIONS))
# # select score for best C
mean_scores = mean_scores.max(axis=0)
bar_offsets = (np.arange(len(N_FEATURES_OPTIONS)) *
               (1 + 1) + .5)

# Draw learning curve
# drawLearningCurve(clf, X_train_std, X_test_std, y_train, y_test, min_size=100, numpoints=50)
# plt.savefig('PCA Learning Curve.png', bbox_inches='tight')

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

# Show learning curve
test = [x[0] for x in clf.grid_scores_]
scoreplot = plt.subplot()
for ind in N_FEATURES_OPTIONS:
    print('Score:' + str(scores[ind-2]))
    print('N_Features:' + str(N_FEATURES_OPTIONS[ind - 2]))
    #scoreplot.scatter(N_FEATURES_OPTIONS[ind-2], scores[ind-2])
    scoreplot.plot(N_FEATURES_OPTIONS[ind-2],
                   scores[ind - 2],
                 'bo',
                   N_FEATURES_OPTIONS[ind - 2],
                   scores[ind - 2],
                 'k')
#scoreplot.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=2)
scoreplot.set_xlabel('Number of Components')
scoreplot.set_ylabel('Mean score')
# scoreplot.set_ylim([0, 1])
scoreplot.set_title('Validation Curve')
#plt.savefig('PCA - SVM Validation Curve.png', bbox_inches='tight')
plt.savefig('Truncated SVD - SVM Validation Curve.png')

Stop_Timer(start_time)