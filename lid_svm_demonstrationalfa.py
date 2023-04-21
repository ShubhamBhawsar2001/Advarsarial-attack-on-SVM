import random
import pandas as pd
from sklearn.model_selection import train_test_split
import alfaflip
import scipy.io as sio
from scipy import optimize
from sklearn import svm
from sklearn.metrics import accuracy_score
from utilities import *

from scipy.optimize import linprog
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC

frac=[0.1,0.2,.3,.4,.5,.6,.7,.8,.9,.99]
noatr=[]
noatt=[]
att=[]
atr=[]
detr=[]
dett=[]
for f in frac:
    seed = 123
    random.seed(seed)
    np.random.seed(seed)

    # Variables
    C = 2.5  # SVM regularization parameter
    GAMMA = 1.5  # gamma of the SVM
    DATASET = 'toy_2'
    ATTACK = 'ALFA'

    # find gamma with the highest KL divergence - offline
    LID_GAMMA = 0.012
    BATCH_SIZE = 50
    ###########

    XMIN = 0
    XMAX = 6
    YMIN = 0
    YMAX = 1.2
    WEIGHT_LB = 0.1

   
    



    # Read the CSV file
    data = pd.read_csv('D:/iisc/sem2/ML/project/lid-svm-master/lid-svm-master/sic_data.csv')
    datacopy=data
    X_data = data.drop('Y', axis=1)  
    y = data['Y']
    xtr, xtt, ytr, ytt = train_test_split(X_data, y, test_size=0.2, random_state=42)
    xtr1, xtt1, ytr1, ytt1=xtr, xtt, ytr, ytt
 

    X = xtr
    y = ytr
    row_count = len(X)

    n_sv = []

    ###############################################################
    # No attack
    ###############################################################
    model = svm.SVC(kernel='rbf', gamma=GAMMA, C=C)
    model.fit(X, y.ravel())

    y_hat2 = model.predict(xtr)
    print("no attack training",accuracy_score(y.ravel(),y_hat2))
    noatr.append(accuracy_score(y.ravel(),y_hat2))
    y_hat = model.predict(xtt)

    print("no attack test",accuracy_score(ytt,y_hat))
    noatt.append(accuracy_score(ytt,y_hat))
    X1=np.array(X)
    ytrr=np.array(ytr).reshape(len(ytr),1)
   # draw_contours(X1, ytrr, model, [], 'SVM - no attack', 'no_attack')

    ###############################################################
    # Attack
    ###############################################################

    X_train=xtr.values.tolist()
    new_y_train=alfaflip.fliping(X_train, xtt, ytr, ytt,f)

    fl=np.array(new_y_train)
    y = fl

    print(ytr.shape)
    print(y.shape)
    fi = np.where(ytr != y)[0]
    


    model = svm.SVC(kernel='rbf', gamma=GAMMA, C=C)
    model.fit(X, y.ravel())
    y_hat2 = model.predict(xtr)
    print("with attack training",accuracy_score(y.ravel(),y_hat2))
    atr.append(accuracy_score(y.ravel(),y_hat2))
    y_hat = model.predict(xtt)
    print("with attack test",accuracy_score(ytt,y_hat))
    att.append(accuracy_score(ytt,y_hat))
    X1=np.array(X)
    ytrr=np.array(ytr).reshape(len(ytr),1)
    fi=fi.reshape(len(fi),1)
    #draw_contours(X1, ytrr, model, fi, 'SVM - alfa attack', 'alfa_attack')

    ###############################################################
    # Test defense - LID
    ###############################################################
    xtr = np.array(xtr)
    ytr=np.array(ytr).reshape(len(ytr),1)
    xtt = np.array(xtt)
    ytt=np.array(ytt).reshape(len(ytt),1)




    pos_density_normal = None
    pos_density_fl = None
    neg_density_normal = None
    neg_density_fl = None





    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y== -1)[0]

    pos_data = xtr[pos_indices, :]
    neg_data = xtr[neg_indices, :]

    pos_lids = get_lids_random_batch(pos_data, LID_GAMMA, lid_type='kernel', k=10, batch_size=BATCH_SIZE)
    neg_lids = get_lids_random_batch(neg_data, LID_GAMMA, lid_type='kernel', k=10, batch_size=BATCH_SIZE)

    # initialize the weights to 0
    lids = np.zeros(row_count)
    # insert LIDs at the correct index
    lids[pos_indices] = pos_lids
    lids[neg_indices] = neg_lids

    # LIDs w.r.t to the opposite class
    pos_lids_opp = get_cross_lids_random_batch(pos_data, neg_data, LID_GAMMA, lid_type='kernel', k=10,
                                            batch_size=BATCH_SIZE)
    neg_lids_opp = get_cross_lids_random_batch(neg_data, pos_data, LID_GAMMA, lid_type='kernel', k=10,
                                            batch_size=BATCH_SIZE)
    # Cross LID values

    pos_cross_lids = np.divide(pos_lids, pos_lids_opp)
    neg_cross_lids = np.divide(neg_lids, neg_lids_opp)

    #
    lids_opp = np.zeros(len(X))
    lids_cross = np.zeros(len(X))
    lids_opp[pos_indices] = pos_lids_opp
    lids_opp[neg_indices] = neg_lids_opp

    lids_cross[pos_indices] = pos_cross_lids
    lids_cross[neg_indices] = neg_cross_lids
    original_lids = lids
    lids = lids_cross

    fi = fi[:, 0]
    # get the indices of rows that are not flipped
    normal_idx = list(set(range(row_count)) - set(fi))

    fl_lid_values = lids[fi]
    normal_lid_values = lids[normal_idx]
    flipped_labels = y[fi]
    normal_labels = np.squeeze(y[normal_idx])

    pos_indices = np.where(normal_labels == 1)[0]
    neg_indices = np.where(normal_labels == -1)[0]

    pos_normal_lids = normal_lid_values[pos_indices]
    neg_normal_lids = normal_lid_values[neg_indices]

    pos_fl_indices = np.where(flipped_labels == 1)[0]
    neg_fl_indices = np.where(flipped_labels == -1)[0]

    pos_fl_lids = fl_lid_values[pos_fl_indices]
    neg_fl_lids = fl_lid_values[neg_fl_indices]

    # placeholders
    weights_pos_normal = np.ones((len(pos_normal_lids),))
    weights_pos_fl = np.ones((len(pos_fl_lids),))
    weights_neg_normal = np.ones((len(neg_normal_lids),))
    weights_neg_fl = np.ones((len(neg_fl_lids),))

    # If there are labels flipped to positive
    if pos_fl_lids.size > 1:
        pos_density_normal = get_kde(pos_normal_lids, bw=0.2)
        pos_density_fl = get_kde(pos_fl_lids, bw=0.1)

        lr_pos_normal, lr_pos_fl = weight_calculation(pos_normal_lids, pos_fl_lids, pos_density_normal,
                                                    pos_density_fl,
                                                    WEIGHT_LB)

        tmp_lid_values = np.concatenate((pos_normal_lids, pos_fl_lids), axis=0)
        tmp_lr = np.concatenate((lr_pos_normal, lr_pos_fl), axis=0)

        # fit a tanh function
        params, params_covariance = optimize.curve_fit(tanh_func, tmp_lid_values, tmp_lr)

        # obtain the weights from the fitted function
        weights_pos_normal = tanh_func(pos_normal_lids, params[0], params[1])
        weights_pos_fl = tanh_func(pos_fl_lids, params[0], params[1])

    if neg_fl_lids.size > 1:
        neg_density_normal = get_kde(neg_normal_lids, bw=0.2)
        neg_density_fl = get_kde(neg_fl_lids, bw=0.1)

        lr_neg_normal, lr_neg_fl = weight_calculation(neg_normal_lids, neg_fl_lids, neg_density_normal,
                                                    neg_density_fl,
                                                    WEIGHT_LB)

        tmp_lid_values = np.concatenate((neg_normal_lids, neg_fl_lids), axis=0)
        tmp_lr = np.concatenate((lr_neg_normal, lr_neg_fl), axis=0)

        params, params_covariance = optimize.curve_fit(tanh_func, tmp_lid_values, tmp_lr)

        weights_neg_normal = tanh_func(neg_normal_lids, params[0], params[1])
        weights_neg_fl = tanh_func(neg_fl_lids, params[0], params[1])

    weights_fl = np.zeros((len(fl_lid_values),))
    weights_fl[pos_fl_indices] = weights_pos_fl
    weights_fl[neg_fl_indices] = weights_neg_fl

    weights_normal = np.zeros((len(normal_lid_values),))
    weights_normal[pos_indices] = weights_pos_normal
    weights_normal[neg_indices] = weights_neg_normal

    weights = np.zeros((row_count,))
    weights[fi] = weights_fl
    weights[normal_idx] = weights_normal
    xtr, xtt, ytr, ytt=xtr1, xtt1, ytr1, ytt1
    model = svm.SVC(kernel='rbf', gamma=GAMMA, C=C)

    model.fit(X, y.ravel(), sample_weight=weights)

    y_hat2 = model.predict(xtr)

    print("with defence test",accuracy_score(y.ravel(),y_hat2))
    detr.append(accuracy_score(y.ravel(),y_hat2))
    y_hat = model.predict(xtt)

    print("with defence test",accuracy_score(ytt,y_hat))
    dett.append(accuracy_score(ytt,y_hat))
    X1=np.array(X)
    ytrr=np.array(ytr).reshape(len(ytr),1)
    fi=fi.reshape(len(fi),1)
    #draw_contours(X1, ytrr, model, fi, 'LID-SVM - alfa attack', 'lid_svm')

print(noatr,atr,detr)
plt.plot(frac, noatr,label='no-attack',color='yellow')

# Plot another line on the same chart/graph
plt.plot(frac, atr,label='attack',color='red')
plt.plot(frac, detr,label='defence',color='blue')
plt.xlabel('fraction of flip')
plt.ylabel('training accuracy')
plt.title('flip(alfa) vs accuracy')
plt.legend()
plt.show()
plt.plot(frac, noatt,label='no-attack',color='yellow')

# Plot another line on the same chart/graph
plt.plot(frac, att,label='attack',color='red')
plt.plot(frac, dett,label='defence',color='blue')
plt.xlabel('fraction of flip')
plt.ylabel('test accuracy')
plt.title('flip(alfa) vs accuracy')
plt.legend()
plt.show()