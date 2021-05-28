import numpy as np
import matplotlib.pyplot as plt


def log_likelihood(W, X, Y):
    N,M = X.shape
    Class = len(np.unique(Y))
    log_like = 0
    for i in range(N):
        for k in range(Class):
            if Y[i]-1 == k:
                if k == Class-1:
                    log_like += np.log(1 / np.sum(np.exp(X[i] @ W)))
                else:
                    log_like += np.log(np.exp(X[i] @ W[:,k]) / np.sum(np.exp(X[i] @ W)))
    return log_like
    
def calculate_gradient(W, X, Y):
    N,M = X.shape
    Class = len(np.unique(Y))
    dw = np.zeros((M,Class))
    temp_X = np.zeros_like(X)
    for m in range(Class-1):
        for i in range(N):
            I = 1 if (Y[i]-1) == m else 0
            temp_X = I - (np.exp(X[i] @ W[:,m]) / np.sum(np.exp(X[i] @ W)))
            dw[:,m] += X[i] * temp_X
    return dw

def gradient_ascent(W, X, Y, test_X, test_Y, alpha = 0.0005):
    iter = 0
    loss = log_likelihood(W, X, Y)
    delta_loss = np.Infinity
    while iter < 1000 and abs(delta_loss) > 1e-15:
        iter += 1
        dw = calculate_gradient(W,X,Y)
        W += alpha * dw
        loss_new = log_likelihood(W, X, Y)
        delta_loss = loss - loss_new
        #if iter % 20 == 0:
        #    acc = accuracy(W,test_X,test_Y,Class) 
        
    return W

def accuracy(W,X,Y,Class):
    N, M  = X.shape
    p = np.zeros((N,Class))
    for i in range(N):
        for k in range(Class):
            if k == Class-1:
                p[i,k] = 1 / np.sum(np.exp(X[i] @ W))
            else:
                p[i,k] = np.exp(X[i] @ W[:,k]) / np.sum(np.exp(X[i] @ W))

    pindx = np.argmax(p, axis = 1)
    pindx +=1
    same_num = np.sum(pindx == (Y.flatten()))
    rate = same_num / Y.shape[0]
    return rate

def main():
    # Load data
    q2_data = np.load('q2_data.npz')
    q2x_train = q2_data["q2x_train"]
    q2y_train = q2_data["q2y_train"]
    q2x_test = q2_data["q2x_test"]
    q2y_test = q2_data["q2y_test"]

    N, M = q2x_train.shape
    Class = len(np.unique(q2y_train))
    W = np.random.random((M,Class - 1))
    W = np.concatenate((W, np.zeros((W.shape[0],1))), axis = 1)
    result_W = gradient_ascent(W, q2x_train, q2y_train, q2x_test, q2y_test)
    acc = accuracy(result_W,q2x_test,q2y_test,Class)
    print(f'The accuracy is {100*acc}%')
    #print(result_W)
    
if __name__ == "__main__":
    main()