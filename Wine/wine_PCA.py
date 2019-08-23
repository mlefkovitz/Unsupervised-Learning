from sklearn.decomposition import PCA
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

# Scale the data
scaler = StandardScaler()
scaler.fit(X)
X_train_std = scaler.transform(X)
X_test_std = scaler.transform(X)
X_toTransform = X_train_std
y_inputs = y

######
# Run PCA with 1:n components
######
reconstructionErrors = []
maxComponents = 46
minComponents = 1
for i in range(minComponents, maxComponents):
    pca = PCA(n_components=i)
    pca.fit(X_toTransform)
    reconstructionErrors.append(reconstructionError(pca,X_toTransform))
    # print diagnostics
    #print('Components \n', projection.components_)
    # print('Number of Components ',pca.n_components_)
    # print('Reconstruction Error ',reconstructionError(pca,X_toTransform))

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
    plt.title('PCA - Error by Number of Components')
    plt.tight_layout()
    #plt.show()
    plt.savefig('PCA Reconstruction Error.png', bbox_inches='tight')

######
# Run PCA Analysis with max components
######
pca = PCA(n_components=46)
pca.fit(X_toTransform)

# print diagnostics
# print('Components \n', pca.components_)
# print('Number of Components \n',pca.n_components_)
# print('Mean \n',pca.mean_)
# print('Expected Variance \n',pca.explained_variance_)
# print('Expected Variance Ratio \n', pca.explained_variance_ratio_)

# Plot histograms
tot = sum(pca.explained_variance_)
var_exp = [(i / tot)*100 for i in sorted(pca.explained_variance_, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
# print(var_exp)
# print(cum_var_exp)
with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))

    plt.bar(range(46), var_exp, alpha=0.5, align='center',
            label='individual explained variance')
    plt.step(range(46), cum_var_exp, where='mid',
             label='cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.xlim(-0.5,46.5)
    plt.legend(loc='best')
    plt.title('PCA Explained Variance Histogram')
    plt.tight_layout()
    #plt.show()
    plt.savefig('PCA Explained Variance Histogram.png', bbox_inches='tight')

# Select interesting components
print('With 2 components, PCA explains %.2f' % cum_var_exp[1], '% of the variance')
print('With 3 components, PCA explains %.2f' % cum_var_exp[2], '% of the variance')
print('50% of variance explained at', np.argmax(cum_var_exp>=50), 'components')
print('80% of variance explained at', np.argmax(cum_var_exp>=80), 'components')
print('90% of variance explained at', np.argmax(cum_var_exp>=90), 'components')
print('99% of variance explained at', np.argmax(cum_var_exp>=99), 'components')

######
# Run PCA Analysis with 2 components and plot
######
pca = PCA(n_components=2)
pca.fit(X_toTransform)

# print diagnostics
# print('Components \n', pca.components_)
# print('Number of Components \n',pca.n_components_)
# print('Mean \n',pca.mean_)
# print('Expected Variance \n',pca.explained_variance_)
# print('Expected Variance Ratio \n', pca.explained_variance_ratio_)
# print(X_toTransform.shape)
# print(np.transpose(pca.components_).shape)

X_transformed = np.dot(X_toTransform, np.transpose(pca.components_))
# print(X_transformed.shape)

# Save plot
with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))
    for yval, lab, col in zip((0, 1),
                              ('<=50K', '>50K'),
                              ('blue', 'red')):
        plt.scatter(X_transformed[y_inputs==yval,0],
                    X_transformed[y_inputs==yval,1],
                    label=lab,
                    c=col)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(loc='best')
    plt.title('PCA 2-D Subspace')
    plt.tight_layout()
    #plt.show()
    plt.savefig('PCA 2D Subspace.png', bbox_inches='tight')

######
# Run PCA Analysis with 3 components and plot
######
pca = PCA(n_components=3)
pca.fit(X_toTransform)

# print diagnostics
# print('Components \n', pca.components_)
# print('Number of Components \n',pca.n_components_)
# print('Mean \n',pca.mean_)
# print('Expected Variance \n',pca.explained_variance_)
# print('Expected Variance Ratio \n', pca.explained_variance_ratio_)
#
# print(X_toTransform.shape)
# print(np.transpose(pca.components_).shape)

X_transformed = np.dot(X_toTransform, np.transpose(pca.components_))
# print(X_transformed.shape)

# Save plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for yval, lab, col in zip((0, 1),
                       ('<=50K', '>50K'),
                    ('blue', 'red')):
    ax.scatter(X_transformed[y_inputs==yval,0],
                X_transformed[y_inputs==yval,1],
                X_transformed[y_inputs == yval, 2],
                label=lab,
                c=col)
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.legend(loc='best')
ax.set_title('PCA 3-D Subspace')
#plt.show()
plt.savefig('PCA 3D Subspace.png', bbox_inches='tight')

Stop_Timer(start_time)