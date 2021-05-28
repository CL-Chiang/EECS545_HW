import numpy as np
import matplotlib.pyplot as plt
from decimal import *

plt.rcParams.update({'font.size': 22})
q2x = np.load('q2x.npy')
q2y = np.load('q2y.npy')
N = len(q2x)
x_poly = np.column_stack([np.ones((N,1)), q2x])

#q2(d)i
def unweighted_linear_regression(x_poly, y):
    xTx = np.dot(np.transpose(x_poly), x_poly)
    w = np.dot(np.dot(np.linalg.inv(xTx), np.transpose(x_poly)), y)
    return w

def plot_unweighted(w, x_poly, x, y):
    fig = plt.figure(0)
    fig.set_size_inches(18.5, 10.5, forward=True)
    fig.canvas.set_window_title('q2(d) i.')  
    pred = np.dot(x_poly, np.transpose(w))
    plt.plot(x, y, 'ob')
    plt.plot(x, pred,'-g')
    plt.legend(('point(x,y)', 'unwieighted'), shadow=True, loc=(0.01, 0.48), handlelength=1.5, fontsize=16)
    plt.xlabel('X')
    plt.ylabel('Y')
    
w = unweighted_linear_regression(x_poly, q2y)
plot_unweighted(w, x_poly, q2x, q2y)

#q2(d)ii
# w = (XTRX)^-1 * XTRY
def weighted_linear_regression(x_query, x_poly, x, y, tau):
    query_len = len(x_query)
    W = []
    Pred = []
    for i in range(query_len):
        r = np.exp(-np.square((x-x_query[i])) / (2 * tau ** 2))
        R = np.diag(r)
        w = np.dot(np.dot(np.dot(np.linalg.inv(np.dot(np.dot(np.transpose(x_poly), R), x_poly)), np.transpose(x_poly)), R), y)
        W.append(w)
        x_query_poly = np.array([1, x_query[i]])
        pred = np.dot(x_query_poly, np.transpose(w))
        
        Pred.append(pred)
        
    return W, Pred

def plot_query_weighted(Pred, x_query, x, y):
    fig = plt.figure(1)
    fig.set_size_inches(18.5, 10.5, forward=True)
    fig.canvas.set_window_title('q2(d) ii.')  
    plt.plot(x, y, 'ob')
    plt.plot(x_query, Pred, '-g')
    plt.legend(('point(x,y)', 'τ = 0.8'), shadow=True, loc=(0.01, 0.48), handlelength=1.5, fontsize=16)
    plt.xlabel('X')
    plt.ylabel('Y')

#print(f'The closed form coefficients for {lamd} : \n {w_r}')
q2x_max = np.max(q2x)
q2x_min = np.min(q2x)
x_query = np.linspace(q2x_min,q2x_max,50)

W0, Pred = weighted_linear_regression(x_query, x_poly, q2x, q2y, 0.8)
plot_query_weighted(Pred, x_query, q2x, q2y)

##q2(d) iii
def plot_multiple_query_weighted(Pred0, Pred1, Pred2, Pred3, x_query, x, y):
    fig = plt.figure(4)
    fig.set_size_inches(18.5, 10.5, forward=True)
    fig.canvas.set_window_title('q2(d) iii.')  
    plt.plot(x, y, 'ob')
    plt.plot(x_query, Pred0, '-g')
    plt.plot(x_query, Pred1, '-r')
    plt.plot(x_query, Pred2, '-b')
    plt.plot(x_query, Pred3, '-m')
    plt.legend(('point(x,y)', 'τ = 0.1', 'τ = 0.3', 'τ = 2', 'τ = 10'), shadow=True, loc=(0.01, 0.48), handlelength=1.5, fontsize=16)
    plt.xlabel('X')
    plt.ylabel('Y')
    print(f'q2 (d) iii:\n When τ is small, the fitting curve followed the data points closely. On the other hand, the curve did not follow closely to data points when τ is large')
W0, Pred0 = weighted_linear_regression(x_query, x_poly, q2x, q2y, 0.1)
W1, Pred1 = weighted_linear_regression(x_query, x_poly, q2x, q2y, 0.3)
W2, Pred2 = weighted_linear_regression(x_query, x_poly, q2x, q2y, 2)
W3, Pred3 = weighted_linear_regression(x_query, x_poly, q2x, q2y, 10)
plot_multiple_query_weighted(Pred0, Pred1, Pred2, Pred3, x_query, q2x, q2y)

plt.show()
