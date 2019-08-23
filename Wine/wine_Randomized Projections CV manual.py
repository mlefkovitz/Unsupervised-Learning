from sklearn.random_projection import GaussianRandomProjection as ProjectionAlgorithm
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
from wine_data import X, y, X_train,  X_test, y_train, y_test

######
# Run Grid Search to find optimal components
######
# import packages
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
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
# svm = SVC(random_state=1)
# parameters = {'kernel':(['linear'])
#              ,'C':[10]
#              ,'gamma':([0.1])
#              }
# clf = GridSearchCV(svm, param_grid=parameters, cv=3)
# N_FEATURES_OPTIONS = [2]

# Define the classifier
nn = MLPClassifier(solver='lbfgs', random_state=1, alpha=0.005, hidden_layer_sizes=3)
grid_params = {'alpha': [0.005]
              ,'hidden_layer_sizes': [3]
              }
clf = GridSearchCV(nn, param_grid=grid_params,cv=3)

# Define the classifier
validationScores = []
trainScores = []
testScores = []
maxComponents = 11
minComponents = 2
for i in range(minComponents, maxComponents):
    projection = ProjectionAlgorithm(n_components=i)
    print('n_components = ', i)
    X_transformed = projection.fit_transform(X_train_std)
    X_testTransformed = projection.transform(X_test_std)
    # print('training...')
    clf.fit(X_transformed, y_train)
    # Identify training and test accuracy
    # print('predicting...')
    y_pred = clf.predict(X_testTransformed)
    print('Misclassified samples: %d' % (y_test != y_pred).sum())
    y_pred_train = clf.predict(X_transformed)
    y_pred_test = clf.predict(X_testTransformed)
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    print('NN train/test accuracy: %.3f/%.3f' % (train_accuracy, test_accuracy))
    print('Validation score: ', clf.best_score_)
    validationScores.append(clf.best_score_)
    trainScores.append(train_accuracy)
    testScores.append(test_accuracy)

print('Max validation score: ', np.max(validationScores), 'with ', np.argmax(validationScores==np.max(validationScores))+2,'clusters')
print('Max train score: ', np.max(trainScores), 'with ', np.argmax(trainScores==np.max(trainScores))+2,'clusters')
print('Max test score: ', np.max(testScores), 'with ', np.argmax(testScores==np.max(testScores))+2,'clusters')

# Draw learning curve
# drawLearningCurve(clf, X_train_std, X_test_std, y_train, y_test, min_size=100, numpoints=50)
# plt.savefig('PCA Learning Curve.png', bbox_inches='tight')

# Show learning curve
test = [x[0] for x in clf.grid_scores_]
scoreplot = plt.subplot()
for ind in range(minComponents, maxComponents):
    # print('Score:' + str(validationScores[ind-2]))
    # print('N_Features:' + str(range(minComponents, maxComponents)[ind - 2]))
    scoreplot.plot(range(minComponents, maxComponents)[ind-2],
                   validationScores[ind - 2],
                 'bo',
                   range(minComponents, maxComponents)[ind - 2],
                   validationScores[ind - 2],
                 'k')
#scoreplot.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=2)
scoreplot.set_xlabel('Number of Components')
scoreplot.set_ylabel('Mean score')
# scoreplot.set_ylim([0, 1])
scoreplot.set_title('Validation Curve')
plt.savefig('RCA - SVM Validation Curve.png')

Stop_Timer(start_time)