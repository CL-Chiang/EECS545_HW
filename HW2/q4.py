import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 22})

def readMatrix(file):
    # Use the code below to read files
    fd = open(file, 'r')
    hdr = fd.readline()
    rows, cols = [int(s) for s in fd.readline().strip().split()]
    tokens = fd.readline().strip().split()
    matrix = np.zeros((rows, cols))
    Y = []
    for i, line in enumerate(fd):
        nums = [int(x) for x in line.strip().split()]
        Y.append(nums[0])
        kv = np.array(nums[1:])
        k = np.cumsum(kv[:-1:2])
        v = kv[1::2]
        matrix[i, k] = v
    return matrix, tokens, np.array(Y)

def nb_train(matrix, category):
    # Implement your algorithm and return 
    state = {}
    N = matrix.shape[1]

    ############################
    # Implement your code here #
    P_spam = np.sum(category) / category.shape[0]
    spam_indx = np.where(category == 1)
    nonspam_indx = np.where(category == 0)
    N_s_j = np.sum(matrix[spam_indx], axis = 0)
    N_ns_j = np.sum(matrix[nonspam_indx], axis = 0)
    P_wordj_given_spam = (N_s_j + 1 ) / ( np.sum(N_s_j)+ N)
    P_wordj_given_nonspam = (N_ns_j + 1 ) / ( np.sum(N_ns_j)+ N)
    state['P_spam'] = P_spam
    state['P_wordj_given_spam'] = P_wordj_given_spam
    state['P_wordj_given_nonspam'] = P_wordj_given_nonspam
    ############################
    
    return state

def nb_test(matrix, state):
    # Classify each email in the test set (each row of the document matrix) as 1 for SPAM and 0 for NON-SPAM
    output = np.zeros(matrix.shape[0])
    
    ############################
    # Implement your code here #
    P_spam = state['P_spam']
    P_wordj_given_spam = state['P_wordj_given_spam']
    P_wordj_given_nonspam = state['P_wordj_given_nonspam']
    for i in range(matrix.shape[0]):
        P_1 = np.log(P_spam) + np.sum(np.dot(np.log(P_wordj_given_spam), matrix[i]))
        P_0 = np.log((1 - P_spam)) + np.sum(np.dot(np.log(P_wordj_given_nonspam), matrix[i]))
        output[i] = 1 if P_1 > P_0 else 0
    ############################
    
    return output

def evaluate(output, label):
    # Use the code below to obtain the accuracy of your algorithm
    error = (output != label).sum() * 1. / len(output)
    print('Error: {:2.4f}%'.format(100*error))

def find_most_indicative_token(state,tokenlist):
    P_wordj_given_spam = state['P_wordj_given_spam']
    P_wordj_given_nonspam = state['P_wordj_given_nonspam']
    log_p = np.log(P_wordj_given_spam/P_wordj_given_nonspam)
    index = np.argpartition(log_p, -5)[-5:]
    print(f'The 5 most indicative tokens are at: {tokenlist[index[0]]}, {tokenlist[index[1]]}, {tokenlist[index[2]]}, {tokenlist[index[3]]}, {tokenlist[index[4]]}')

def nb_train_test_diff_size(matrix, category, X_test, Y_test):
    #repeate train with different size
    sample_size = matrix.shape[0]
    partition = 100
    error = np.zeros(partition)
    for i in range(partition):
        size = int((i+1) * sample_size/ partition)
        X_train = matrix[:size,:]
        Y_train = category[:size]
        state = nb_train(X_train, Y_train)
        prediction = nb_test(X_test, state)
        error[i] = (prediction != Y_test).sum() * 1. / len(prediction)
        #print(f'error{i}: {100 * error[i]}%')
    plot_q4c(error)

def plot_q4c(error):
    N = len(error)
    x_cord = np.arange(0, 100, int(100/N))
    x_cord += int(100/N)
    fig = plt.figure(0)
    fig.set_size_inches(18.5, 10.5, forward=True)
    fig.canvas.set_window_title('q4(c)')
    plt.plot(x_cord,error, 'bo-')
    plt.xlabel('using percent of the total training dataset(%)')
    plt.ylabel('error rate(%)')
    plt.show()

def main():
    # Note1: tokenlists (list of all tokens) from MATRIX.TRAIN and MATRIX.TEST are identical
    # Note2: Spam emails are denoted as class 1, and non-spam ones as class 0.
    # Note3: The shape of the data matrix (document matrix): (number of emails) by (number of tokens)

    # Load files
    dataMatrix_train, tokenlist, category_train = readMatrix('q4_data/MATRIX.TRAIN')
    dataMatrix_test, tokenlist, category_test = readMatrix('q4_data/MATRIX.TEST')

    # Train
    state = nb_train(dataMatrix_train, category_train)

    # Test and evluate
    prediction = nb_test(dataMatrix_test, state)
    evaluate(prediction, category_test)

    # find the 5 most indicative tokens
    find_most_indicative_token(state, tokenlist)

    nb_train_test_diff_size(dataMatrix_train, category_train, dataMatrix_test, category_test)






if __name__ == "__main__":
    main()
        