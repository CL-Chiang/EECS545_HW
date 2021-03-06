import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 22})
q1xTrain = np.load('q1xTrain.npy')
q1yTrain = np.load('q1yTrain.npy')
q1xTest = np.load('q1xTest.npy')
q1yTest = np.load('q1yTest.npy')
#print(q1xTrain)
#print(q1yTrain)


# q1 (a)
def batch_gradienct_descent(iter, x, y, alpha = 0.05, e = 0.2):
    # y = ax + b
    costBGD = []
    a = 0
    b = 0
    num = len(y)
    First = 1
    converge_i = 0
    for i in range(iter):
        loss = a * x + b - y
        cost = np.sum(loss**2)/ num
        gradb = np.sum(loss) / num
        grada = np.sum(loss * x) / num
        a -= alpha * grada
        b -= alpha * gradb
        costBGD.append(cost)
        if cost < e and First ==1:
            First = 0
            converge_i = i
    return a, b, costBGD, converge_i

def stochastic_gradient_descent(iter, x, y, alpha_ub = 0.05, alpha_lb = 0.005, e = 0.2):
    # y = ax + b
    costSGD = []
    a = 0
    b = 0
    num = len(y)
    First = 1
    converge_i = 0
    alpha = alpha_ub
    alpha_grad = (alpha_lb - alpha_ub) / iter
    for i in range(iter):
        cost = 0
        alpha += alpha_grad
        for j in range(num):
            rndindx = np.random.randint(num)
            xj = x[rndindx]
            yj = y[rndindx]
            loss = a * xj + b - yj
            gradb = loss / num
            grada = loss * xj /num
            a -= alpha * grada
            b -= alpha * gradb
            cost += loss**2
        cost /= num
        if cost < e and First ==1:
            First =0
            converge_i = i
        costSGD.append(cost)

    return a, b, costSGD, converge_i

def plot_print_q1a(w0_BGD, w1_BGD, w0_SGD, w1_SGD, i_BGD, i_SGD):
    print(f'q1 (a):')
    print(f'The coefficients generated by batch gradient descent is:  {w0_BGD} , {w1_BGD}')
    print(f'The coefficients generated by stochastic gradient descent is:  {w0_SGD} , {w1_SGD}')
    print(f'SGD converged faster because SGD converged at {i_SGD}-th iteration while BGD converged at {i_BGD}-th iteration')
    fig = plt.figure(0)
    fig.set_size_inches(18.5, 10.5, forward=True)
    fig.canvas.set_window_title('q1(a)')
    plt.suptitle('Mean Squared Error')
    plt.subplot(2, 1, 1)
    plt.subplots_adjust(hspace = 0.5) 
    plt.title('BGD')
    plt.plot(costBGD)
    plt.xlabel('iteration')
    plt.ylabel('E_RMS')
    plt.subplot(2, 1, 2)
    plt.title('SGD')
    plt.plot(costSGD)
    plt.xlabel('iteration')
    plt.ylabel('E_RMS')
    #plt.show()
    
w0_BGD , w1_BGD, costBGD, i_BGD = batch_gradienct_descent(2000,q1xTrain,q1yTrain)
w0_SGD , w1_SGD, costSGD, i_SGD = stochastic_gradient_descent(2000,q1xTrain,q1yTrain)

plot_print_q1a(w0_BGD, w1_BGD, w0_SGD, w1_SGD, i_BGD, i_SGD)



# q1 (b)
def closed_form_solution(X,Y):
    # theta = (XTX)^-1 * XT * Y
    XTX = np.dot(np.transpose(X), X)
    theta = np.dot(np.dot(np.linalg.inv(XTX), np.transpose(X)), Y)
    return theta

def calculate_plot_error(TrainX, TrainY, TestX, TestY):
    N = len(TrainY)
    train_error = np.zeros(10)
    test_error = np.zeros(10)
    for i in range(10):
        if i == 0:
            Xtrain_poly = np.ones((N,1))
            Xtest_poly = np.ones((N,1))
        else:
            Xi_train = TrainX**i
            Xi_test = TestX**i
            Xtrain_poly = np.column_stack([Xtrain_poly, Xi_train])
            Xtest_poly =  np.column_stack([Xtest_poly, Xi_test])
        theta = closed_form_solution(Xtrain_poly, q1yTrain)
        train_erms = np.sqrt(np.sum((np.dot(Xtrain_poly, theta) - q1yTrain)**2) / N)
        train_error[i] = train_erms
        test_erms = np.sqrt(np.sum((np.dot(Xtest_poly, theta) - q1yTest)**2) / N)
        test_error[i] = test_erms
        #print(train_erms)
        #print(test_erms
    return train_error, test_error

def plot_print_q1b(train_error, test_error):
    fig = plt.figure(1)
    fig.set_size_inches(18.5, 10.5, forward=True)
    fig.canvas.set_window_title('q1(b)')    
    total_error = train_error + test_error
    best_degree = np.argmin(total_error)

    print(f'q1 (b):')
    print(f'The {best_degree}-th degree best fits the data since it had the least error')
    print(f'Yes, there was an evidence of overfitting. In plot q1(b), we could see that the training error was the least when M = 9.')
    print(f'However, the testing error was really large when M = 9. That is because the 9-th degree polynomial overfit the training data.')

    plt.xlabel('M')
    plt.ylabel('E_RMS')
    plt.plot(train_error, 'bo-')
    plt.plot(test_error, 'ro-')
    plt.legend(('Training', 'Test'), shadow=True, loc=(0.01, 0.75), handlelength=1.5, fontsize=22)

train_error, test_error = calculate_plot_error(q1xTrain, q1yTrain, q1xTest, q1yTest)
plot_print_q1b(train_error, test_error)


#q1(c)
#theta = (XTX + lambda * I)^-1 * XT * y
def closed_form_solution_lambda(X,Y,lmbda):
    theta_list = []
    #theta = (XTX + lambda * I)^-1 * XT * y
    for i in range(10):
        XTX = np.dot(np.transpose(X), X)
        theta = np.dot(np.dot(np.linalg.inv(XTX + lmbda[i] * np.identity(XTX.shape[0])), np.transpose(X)), Y)
        theta_list.append(theta)
    
    #print(theta_list)
    return theta_list


def calcute_ermc(TrainX, TrainY, TestX, TestY):
    train_error = np.zeros(10)
    test_error = np.zeros(10)
    
    # create lambda list
    np.logspace(0,9,9,base = 1e-8)
    lmbda = np.append(0,np.logspace(-8,0,9,base = 10))
    
    # 9-th polynomial
    N = len(TrainY)
    for i in range(10):
        if i == 0:
            Xtrain_poly = np.ones((N,1))
            Xtest_poly = np.ones((N,1))
        else:
            Xi_train = TrainX**i
            Xi_test = TestX**i
            Xtrain_poly = np.column_stack([Xtrain_poly, Xi_train])
            Xtest_poly =  np.column_stack([Xtest_poly, Xi_test])

    theta_list = closed_form_solution_lambda(Xtrain_poly, q1yTrain, lmbda)
    
    # 10 diff lambda 10 diff erms(train/test)
    for i in range(10):
        theta = theta_list[i]
        train_erms = np.sqrt(np.sum((np.dot(Xtrain_poly, theta) - q1yTrain)**2) / N)
        test_erms = np.sqrt(np.sum((np.dot(Xtest_poly, theta) - q1yTest)**2) / N)
        train_error[i] = train_erms
        test_error[i] = test_erms

    return train_error, test_error, lmbda

def plot_print_q1c(train_error, test_error, lmbda):
    fig = plt.figure(2)
    fig.set_size_inches(18.5, 10.5, forward=True)
    fig.canvas.set_window_title('q1(c)')    
    lmbda[0] += 1e-17 # To avoid ln(0) error
    lmbda = np.log(lmbda)
    plt.plot(lmbda, train_error, 'bo-')
    plt.plot(lmbda, test_error, 'ro-')
    plt.legend(('Training', 'Test'), shadow=True, loc=(0.01, 0.5), handlelength=1.5, fontsize=22)
    plt.xlabel('ln(??)\n In this plot, ?? = 0 is replaced with ?? = 1e-17 because ln(??) would cause error when ?? = 0')
    plt.ylabel('E_RMS')
    total_error = train_error + test_error
    best_lambda_index = np.argmin(total_error)

    best_lambda = 0 if best_lambda_index == 0 else  np.power(10, float(best_lambda_index-9))
    
    print(f'q1 (c):')
    print(f'The ?? seemed to work the best when it equaled to {best_lambda}')
    


train_error, test_error, lmbda = calcute_ermc(q1xTrain, q1yTrain,q1xTest,q1yTest)
plot_print_q1c(train_error, test_error, lmbda)
plt.show()