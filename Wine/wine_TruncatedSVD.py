from sklearn.decomposition import TruncatedSVD as projectionmethod
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
from income_data import X, y, X_train,  X_test, y_train, y_test

# Scale the data
scaler = StandardScaler()
scaler.fit(X)
X_train_std = scaler.transform(X)
X_test_std = scaler.transform(X)
X_toTransform = X_train_std
y_inputs = y

######
# Run Truncated SVD with 1:n components
######
reconstructionErrors = []
maxComponents = 45
minComponents = 1
for i in range(minComponents, maxComponents):
    projection = projectionmethod(n_components=i)
    projection.fit_transform(X_toTransform)
    reconstructionErrors.append(reconstructionError(projection, X_toTransform))
    tot = sum(projection.explained_variance_)
    var_exp = [(j / tot) * 100 for j in sorted(projection.explained_variance_, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)
    # print diagnostics
    print('Number of Components ',i)
    # print('Components \n', projection.components_)
    print('Reconstruction Error ', reconstructionError(projection, X_toTransform))
    print('explained variance \n', var_exp)
    print('cumulative explained variance \n', cum_var_exp)

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
    plt.title('Truncated SVD - Error by Number of Components')
    plt.tight_layout()
    #plt.show()
    plt.savefig('Truncated SVD Reconstruction Error.png', bbox_inches='tight')

######
# Run Truncated SVD with max components
######

projection = projectionmethod(n_components=45)
#ica.fit(X_toTransform)
X_Transformed = projection.fit_transform(X_toTransform)

# print diagnostics
print('Components \n', projection.components_)
print('Component shape \n', projection.components_.shape)
print('X_transformed \n', X_Transformed)
print('X_transformed shape \n', X_Transformed.shape)

# Plot histograms
tot = sum(projection.explained_variance_)
var_exp = [(i / tot)*100 for i in sorted(projection.explained_variance_, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
print(var_exp)
print(cum_var_exp)
with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))

    plt.bar(range(45), var_exp, alpha=0.5, align='center',
            label='individual explained variance')
    plt.step(range(45), cum_var_exp, where='mid',
             label='cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Components')
    plt.xlim(-0.5,45.5)
    plt.legend(loc='best')
    plt.title('Truncated SVD Explained Variance Histogram')
    plt.tight_layout()
    #plt.show()
    plt.savefig('Truncated SVD Explained Variance Histogram.png', bbox_inches='tight')

# Select interesting components
print('With 2 components, PCA explains %.2f' % cum_var_exp[1], '% of the variance')
print('With 3 components, PCA explains %.2f' % cum_var_exp[2], '% of the variance')
print('50% of variance explained at', np.argmax(cum_var_exp>=50), 'components')
print('80% of variance explained at', np.argmax(cum_var_exp>=80), 'components')
print('90% of variance explained at', np.argmax(cum_var_exp>=90), 'components')
print('99% of variance explained at', np.argmax(cum_var_exp>=99), 'components')

######
# Run Truncated SVD Analysis with 2 components and plot
######
projection = projectionmethod(n_components=2)
X_Transformed = projection.fit_transform(X_toTransform)

# print diagnostics
# print('Components \n', ica.components_)
# print('Number of iterations \n',ica.n_iter_)
# print('Mixing \n',ica.mixing_)
# print('S \n',S)

X_transformed = np.dot(X_toTransform, np.transpose(projection.components_))
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
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend(loc='best')
    plt.title('Truncated SVD 2-D Subspace')
    plt.tight_layout()
    #plt.show()
    plt.savefig('Truncated SVD 2D Subspace.png', bbox_inches='tight')

######
# Run Truncated SVD Analysis with 3 components and plot
######
projection = projectionmethod(n_components=3)
X_Transformed = projection.fit_transform(X_toTransform)

# print diagnostics
# print('Components \n', ica.components_)
# print('Number of iterations \n',ica.n_iter_)
# print('Mixing \n',ica.mixing_)
# print('S \n',S)
#
# print(X_toTransform.shape)
# print(np.transpose(ica.components_).shape)

X_transformed = np.dot(X_toTransform, np.transpose(projection.components_))
#print(X_transformed.shape)

# Save plot
fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
ax = Axes3D(fig)

for yval, lab, col in zip((0, 1),
                       ('<=50K', '>50K'),
                    ('blue', 'red')):
    ax.scatter(X_transformed[y_inputs==yval,0],
                X_transformed[y_inputs==yval,1],
                X_transformed[y_inputs == yval, 2],
                label=lab,
                c=col)
ax.set_xlabel('Component 1')
ax.set_ylabel('Component 2')
ax.set_zlabel('Component 3')
ax.legend(loc='best')
ax.set_title('Truncated SVD 3-D Subspace')
#plt.show()
plt.savefig('Truncated SVD 3D Subspace.png', bbox_inches='tight')

Stop_Timer(start_time)
