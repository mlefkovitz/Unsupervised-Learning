from sklearn.decomposition import FastICA
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from Print_Timer_Results import *
from ReconstructionError import *
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import kurtosis
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
X_toTransform = X_train_std
y_inputs = y

######
# Run ICA with 1:n components
######
reconstructionErrors = []
averageKurtosis = []
maxComponents = 41
minComponents = 1
for i in range(minComponents, maxComponents):
    ica = FastICA(n_components=i)
    ica.fit(X_toTransform)
    reconstructionErrors.append(reconstructionError(ica,X_toTransform))
    averageKurtosis.append(np.average(kurtosis(ica.components_)))
    # print diagnostics
    # print('Components \n', ica.components_)
    print('Number of Components ',i)
    print('Number of Iterations ', ica.n_iter_)
    print('Kurtosis ', np.average(kurtosis(ica.components_)))
    print('Reconstruction Error ',reconstructionError(ica,X_toTransform))

# print(reconstructionErrors)
# print('Min reconstruction error: ' % np.max(reconstructionErrors), 'with ', np.max(reconstructionErrors==np.max(reconstructionErrors))+1,'components')

# Save reconstruction error plot
with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))
    plt.plot(range(minComponents,maxComponents),
             reconstructionErrors,
             'bo',
             range(minComponents, maxComponents),
             reconstructionErrors,
             'k')
    plt.xlabel('Number of Components')
    plt.ylabel('Reconstruction Error')
    plt.title('ICA - Error by Number of Components')
    plt.tight_layout()
    #plt.show()
    plt.savefig('ICA Reconstruction Error.png', bbox_inches='tight')

# Save average kurtosis plot
with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))
    plt.plot(range(minComponents,maxComponents),
             averageKurtosis,
             'bo',
             range(minComponents, maxComponents),
             averageKurtosis,
             'k')
    plt.xlabel('Number of Components')
    plt.ylabel('Average Kurtosis')
    plt.title('ICA - Kurtosis by Number of Components')
    plt.tight_layout()
    #plt.show()
    plt.savefig('ICA Kurtosis.png', bbox_inches='tight')

######
# Run ICA with max components
######

ica = FastICA(n_components=46)
#ica.fit(X_toTransform)
X_Transformed = ica.fit_transform(X_toTransform)

# print diagnostics
print('Components \n', ica.components_)
print('Component shape \n', ica.components_.shape)
print('Number of iterations \n',ica.n_iter_)
print('Mixing \n',ica.mixing_)
print('X_transformed \n', X_Transformed)
print('X_transformed shape \n', X_Transformed.shape)
print('Kurtosis ', np.average(kurtosis(ica.components_)))

# ######
# # Run ICA Analysis with 2 components and plot
# ######
# ica = FastICA(n_components=2)
# X_Transformed = ica.fit_transform(X_toTransform)
#
# # print diagnostics
# # print('Components \n', ica.components_)
# # print('Number of iterations \n',ica.n_iter_)
# # print('Mixing \n',ica.mixing_)
# # print('S \n',S)
#
# X_transformed = np.dot(X_toTransform, np.transpose(ica.components_))
# # print(X_transformed.shape)
#
# # Save plot
# with plt.style.context('seaborn-whitegrid'):
#     plt.figure(figsize=(6, 4))
#     for yval, lab, col in zip((0, 1),
#                               ('<=50K', '>50K'),
#                               ('blue', 'red')):
#         plt.scatter(X_transformed[y_inputs==yval,0],
#                     X_transformed[y_inputs==yval,1],
#                     label=lab,
#                     c=col)
#     plt.xlabel('Independent Component 1')
#     plt.ylabel('Independent Component 2')
#     plt.legend(loc='best')
#     plt.title('ICA 2-D Subspace')
#     plt.tight_layout()
#     #plt.show()
#     plt.savefig('ICA 2D Subspace.png', bbox_inches='tight')
#
# ######
# # Run ICA Analysis with 3 components and plot
# ######
# ica = FastICA(n_components=3)
# X_Transformed = ica.fit_transform(X_toTransform)
#
# # print diagnostics
# # print('Components \n', ica.components_)
# # print('Number of iterations \n',ica.n_iter_)
# # print('Mixing \n',ica.mixing_)
# # print('S \n',S)
# #
# # print(X_toTransform.shape)
# # print(np.transpose(ica.components_).shape)
#
# X_transformed = np.dot(X_toTransform, np.transpose(ica.components_))
# #print(X_transformed.shape)
#
# # Save plot
# fig = plt.figure()
# #ax = fig.add_subplot(111, projection='3d')
# ax = Axes3D(fig)
#
# for yval, lab, col in zip((0, 1),
#                        ('<=50K', '>50K'),
#                     ('blue', 'red')):
#     ax.scatter(X_transformed[y_inputs==yval,0],
#                 X_transformed[y_inputs==yval,1],
#                 X_transformed[y_inputs == yval, 2],
#                 label=lab,
#                 c=col)
# ax.set_xlabel('Independent Component 1')
# ax.set_ylabel('Independent Component 2')
# ax.set_zlabel('Independent Component 3')
# ax.legend(loc='best')
# ax.set_title('ICA 3-D Subspace')
# #plt.show()
# plt.savefig('ICA 3D Subspace.png', bbox_inches='tight')

Stop_Timer(start_time)
