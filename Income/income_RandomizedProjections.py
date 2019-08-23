from sklearn.random_projection import GaussianRandomProjection as RandomProjection
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

# Scale the data
scaler = StandardScaler()
scaler.fit(X)
X_train_std = scaler.transform(X)
X_test_std = scaler.transform(X)
X_toTransform = X_train_std
y_input = y

######
# Run initial Projection Analysis with 1:n components
######
reconstructionErrors = []
maxComponents = 46
minComponents = 1
for i in range(minComponents, maxComponents):
    projection = RandomProjection(n_components=i)
    projection.fit(X_toTransform)
    reconstructionErrors.append(reconstructionError(projection,X_toTransform))
    # print diagnostics
    #print('Components \n', projection.components_)
    print('Number of Components ',projection.n_components_)
    print('Reconstruction Error ',reconstructionError(projection,X_toTransform))

print(reconstructionErrors)
print('Min reconstruction error: ' % np.max(reconstructionErrors), 'with ', np.max(reconstructionErrors==np.max(reconstructionErrors))+1,'components')

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
    plt.title('Randomized Projections - Error by Number of Components')
    plt.tight_layout()
    #plt.show()
    plt.savefig('Randomized Projections Reconstruction Error.png', bbox_inches='tight')

######
# Run Randomized Projection Analysis with 2 components and plot
######
projection = RandomProjection(n_components=2)
X_transformed = projection.fit_transform(X_toTransform)

# print diagnostics
# print('Components \n', projection.components_)
# print('Number of Components \n',projection.n_components_)
# print(X_toTransform.shape)
# print(X_transformed.shape)

# Save plot
with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))
    for yval, lab, col in zip((0, 1),
                              ('<=50K', '>50K'),
                              ('blue', 'red')):
        plt.scatter(X_transformed[y_input==yval,0],
                    X_transformed[y_input==yval,1],
                    label=lab,
                    c=col)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend(loc='best')
    plt.title('Randomized Projection 2-D Subspace')
    plt.tight_layout()
    #plt.show()
    plt.savefig('Randomized Projection 2D Subspace.png', bbox_inches='tight')

######
# Run Randomized Projection Analysis with 3 components and plot
######
projection = RandomProjection(n_components=3)
X_transformed = projection.fit_transform(X_toTransform)

# print diagnostics
# print('Components \n', projection.components_)
# print('Number of Components \n',projection.n_components_)
# print(X_toTransform.shape)
# print(X_transformed.shape)

# Save plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for yval, lab, col in zip((0, 1),
                       ('<=50K', '>50K'),
                    ('blue', 'red')):
    ax.scatter(X_transformed[y_input==yval,0],
                X_transformed[y_input==yval,1],
                X_transformed[y_input == yval, 2],
                label=lab,
                c=col)
ax.set_xlabel('Component 1')
ax.set_ylabel('Component 2')
ax.set_zlabel('Component 3')
ax.legend(loc='best')
ax.set_title('Randomized Projection 3-D Subspace')
#plt.show()
plt.savefig('Randomized Projection 3D Subspace.png', bbox_inches='tight')

Stop_Timer(start_time)