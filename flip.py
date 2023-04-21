import numpy as np
from sklearn.svm import SVC
def flip_labels(y_train, p):
    y_train_flip = np.copy(y_train)
    num_flip = int(p * len(y_train_flip))
    idx_flip = np.random.choice(len(y_train_flip), size=num_flip, replace=False)
    y_train_flip[idx_flip] = -y_train_flip[idx_flip]
    return y_train_flip
def furthest_first_label_flip(X, y, percent_flip,GAMMA,C):
    # Get the decision function values for the samples
    clf = SVC(kernel='rbf', gamma=GAMMA, C=C)
    clf.fit(X, y)
    decision_values = clf.decision_function(X)
    
    # Get the indices of the samples sorted by their absolute decision function values
    sorted_indices = np.argsort(np.abs(decision_values))
    
    # Determine how many samples to flip labels for
    num_flip = int(percent_flip * len(X))
    
    # Flip the labels for the furthest samples
    flip_indices = sorted_indices[-num_flip:]
    
    y[flip_indices] = -y[flip_indices]
    
    return y,flip_indices
def nearest_first_label_flip(X, y, percent_flip, GAMMA, C):
    # Get the decision function values for the samples
    clf = SVC(kernel='linear')
    clf.fit(X, y)
    decision_values = clf.decision_function(X)

    # Get the distance of each sample to the hyperplane
    distances = np.abs(decision_values) / np.linalg.norm(clf.coef_)

    # Get the indices of the samples sorted by distance to the hyperplane
    sorted_indices = np.argsort(distances)

    # Determine how many samples to flip labels for
    num_flip = int(percent_flip * len(X))

    # Flip the labels for the nearest samples
    flip_indices = sorted_indices[:num_flip]
    y[flip_indices] = -y[flip_indices]
    
    return y,flip_indices

