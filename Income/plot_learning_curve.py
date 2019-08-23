import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# seperating data sets for cross validation

# compute the rms error
def compute_error(x, y, model):
    yfit = model.predict(x)
    rms = np.sqrt(np.mean((y - yfit) ** 2))
    ascore = accuracy_score(y, yfit)
    #return rms
    return ascore


def drawLearningCurve(model, X_train, X_test, y_train, y_test, min_size=1, numpoints=50):
    sizes = np.linspace(min_size, X_train.shape[0], numpoints, endpoint=True).astype(int)
    train_error = np.zeros(sizes.shape)
    test_error = np.zeros(sizes.shape)
    CV_error = np.zeros(sizes.shape)
    UseCV = 0
    if type(model) is GridSearchCV:
        UseCV = 1

    for i, size in enumerate(sizes):
        # getting the predicted results of the GaussianNB
        model.fit(X_train[:size, :], y_train[:size])
        predicted = model.predict(X_train)

        if UseCV == 1:
            CV_error[i] = model.best_score_

        # compute the validation error
        test_error[i] = compute_error(X_test, y_test, model)

        # compute the training error
        train_error[i] = compute_error(X_train[:size, :], y_train[:size], model)

    # draw the plot
    fig, ax = plt.subplots()
    ax.plot(sizes, test_error, lw=2, label='test score', color='green')
    if UseCV == 1:
        ax.plot(sizes, CV_error, lw=2, label='CV score', color='brown')

    ax.plot(sizes, train_error, lw=2, label='training score', color='blue')
    ax.set_xlabel('training examples')
    ax.set_ylabel('accuracy/score')

    ax.legend(loc=0)
    ax.set_xlim(0, X_train.shape[0]+1)
    ax.set_title('Learning Curve')
