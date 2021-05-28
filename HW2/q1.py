import numpy as np
import math
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 22})

def sigmoid(WTX):
    return 1.0 / (1.0 + np.exp(-WTX))

def log_likelihood(W, X, Y):
    WTX = np.dot(X,W)
    sig = sigmoid(WTX)
    print
    return np.sum(Y * np.log(sig) + (1 - Y) * np.log(1 - sig)) 

def Hessian(W, X):
    WTX = np.dot(X,W)
    h = np.multiply(sigmoid(WTX), 1 - sigmoid(WTX))
    D = np.diag(h.flatten())
    return -np.dot(np.dot(np.transpose(X), D), X)


def newtom_method(W, X, Y):
    Y = np.transpose(np.array([Y]))
    Wlist = np.array([W])
    l = log_likelihood(W, X, Y)
    delta_l = np.Infinity
    while abs(delta_l) > 1e-10:
        # calculate grad l
        WTX = np.dot(X,W)
        grad = np.dot(np.transpose(X), (Y-sigmoid(WTX)))        
        # calculate Hessian
        H = Hessian(W,X)
        #calculate step
        step = np.dot(np.linalg.inv(H), grad)
        W = W - step

        # calculate log likelihood
        l_new = log_likelihood(W,X,Y)
        delta_l = l_new - l
        l = l_new
        Wlist = np.append(Wlist, [W], axis=0)

    return W, Wlist

def predict(W,X):
    sig = sigmoid(np.dot(X, W))
    predict_Y = np.copy(sig)
    predict_Y = predict_Y.round()
    return predict_Y
        

def print_q1b(W, Wlist):
    fig = plt.figure(0)
    fig.set_size_inches(18.5, 10.5, forward=True)
    fig.canvas.set_window_title('q1(b)')
    plt.plot(Wlist[:,0], 'o-b')
    plt.plot(Wlist[:,1], 'o-r')
    plt.plot(Wlist[:,2], 'o-g')
    plt.legend(('W0(Intercept)', 'W1', 'W2'), shadow=True, loc=(0.01, 0.01), handlelength=1.5, fontsize=22)
    plt.xlabel('iteration')
    plt.ylabel('W value')
    print(f'The coefficient W is:\n{W}')

def print_q1c(W, X_plt, X, Y):
    fig = plt.figure(1)
    fig.set_size_inches(18.5, 10.5, forward=True)
    fig.canvas.set_window_title('q1(c)')
    index1 = np.where(Y == 1)
    index0 = np.where(Y == 0)
    x1, x2 = np.mgrid[np.min(X_plt[:,0]):np.max(X_plt[:,0]):.01, np.min(X_plt[:,1]):np.max(X_plt[:,1]):.01]
    grid = np.c_[x1.ravel(), x2.ravel()]
    N = grid.shape[0]
    X_contour = np.concatenate((np.ones((N, 1)), grid), axis=1)
    Z = predict(W,X_contour)
    Z = np.array(Z).reshape(x1.shape)
    plot_t = plt.scatter(X_plt[index1[0],0], X_plt[index1[0],1], marker='o')
    plot_f = plt.scatter(X_plt[index0[0],0], X_plt[index0[0],1], marker='x', color = 'r')
    plt.contour(x1, x2, Z, levels=[0.5], cmap='gray')
    plt.legend((plot_t, plot_f), ('predict = 1', 'predict = 0'), shadow=True, loc=(0.01, 0.01), handlelength=1.5, fontsize=22)
    plt.xlabel('x1')
    plt.ylabel('x2')


def main():
    # We format the data matrix so that each row is the feature for one sample.
    # The number of rows is the number of data samples.
    # The number of columns is the dimension of one data sample.
    X = np.load('q1x.npy')
    X_forq1c = X
    N = X.shape[0]
    Y = np.load('q1y.npy')
    # To consider intercept term, we append a column vector with all entries=1.
    # Then the coefficient correpsonding to this column is an intercept term.
    X = np.concatenate((np.ones((N, 1)), X), axis=1)
    # M is dimension of weights
    M = X.shape[1]
    W = np.zeros((M,1))
    W, Wlist = newtom_method(W,X,Y)
    predict_Y = predict(W,X)
    print_q1b(W,Wlist)
    print_q1c(W, X_forq1c, X, Y)
    plt.show()

    

if __name__ == "__main__":
    main()