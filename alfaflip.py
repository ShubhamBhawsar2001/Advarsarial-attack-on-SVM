

from scipy.optimize import linprog
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split




def fliping(X_train, X_test, y_train, y_test,f):
    
     
        def make_meshgrid(x, y, h=0.2):
            x_min, x_max = x.min() - 1, x.max() + 1
            y_min, y_max = y.min() - 1, y.max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                np.arange(y_min, y_max, h))
            return xx, yy

        def plot_contours(clf, xx1,yy, **params):
            Z = clf.predict(np.c_[xx1.ravel(),yy.ravel()])
            Z = Z.reshape(xx1.shape)
            out = plt.contourf(xx1,yy, Z, **params)
            return out

        def SVMPlot(classifier,X_train, y_train):
            X_train = np.array(X_train)
            xx1,yy= make_meshgrid(X_train[:,0], X_train[:,1])
            plot_contours(classifier,xx1,yy,cmap=plt.cm.coolwarm, alpha=0.8)
            plt.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap=plt.cm.coolwarm, s=20, edgecolors='k')

        def solveLP(eps):
            X = X_train
            func_coeff = [0]*(len(U))
            gamma = 1
            C = 100*f
            for i in range(len(U)):
                func_coeff[i] = (eps[i]-psi[i])

            #constraints
            a_ub = []
            b_ub = []
            a_eq = []
            b_eq = []

            temp = [0]*len(U)
            for i in range(len(X), len(U)):
                temp[i] = c[i-len(X)]

            a_ub.append(temp)
            b_ub.append(C)

            for i in range(len(X)):
                temp = [0]*len(U)
                temp[i] = temp[len(X)+i] = 1
                a_eq.append(temp)
                b_eq.append(1)

            q_bound = (0,1)
            Q_bound = tuple([(0,1)]*len(U))

            q = linprog(func_coeff, A_ub = a_ub, b_ub = b_ub, A_eq = a_eq, b_eq = b_eq, bounds = Q_bound, options={"disp": False, "maxiter": 10000}).x
            return q

        def solveQP(q):
            X = X_train
            eps = [0]*len(U)
            new_data = []
            newX = [0]*len(data)
            newY = [0]*len(data)
            k = 0
            for i in range(len(U)):
                if q[i]!=0:
                    newX[k] = [U['X1'][i], U['X2'][i]]
                    newY[k] = U['Y'][i]
                    k+=1
            new_classifier = SVC(gamma='auto')
            new_classifier.fit(newX, newY)    
            for i in range(len(U)):
                xi = np.zeros(2)
                xi[0],xi[1],yi = U.iloc[i]
                new_predicted = new_classifier.predict(xi.reshape(1,-1))
                eps[i] = max(0, 1 - yi*new_predicted)
            return newX, newY, new_classifier, eps

        
        normal_classifier = SVC(gamma='auto')
        
        
        normal_classifier.fit(X_train, y_train)
        data = pd.DataFrame(data = X_train, columns = ['X1', 'X2'])
        data['Y'] = y_train.values

        temp_data = data.copy()
        temp_data['Y']*=-1

        U = pd.DataFrame(np.vstack((data, temp_data)), columns=['X1', 'X2', 'Y'])
        
        psi = [0]*len(U)
        eps = [0]*len(U)
        c   = [1]*len(U)
        q   = [0]*len(U)
        for i in range(len(U)):
            xi = np.zeros(2)
            xi[0],xi[1],yi = U.iloc[i]
            normal_predicted = normal_classifier.predict(xi.reshape(1,-1))              
            psi[i] = max(0, 1 - yi*normal_predicted)
        maxIter = 10
        curIter = 1
        while curIter<=maxIter:
            q = solveLP(eps)
            new_X_train, new_y_train, new_classifier, eps = solveQP(q)
            curIter+=1

        


        return new_y_train
