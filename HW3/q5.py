# EECS 545 HW3 Q5
# Your name: (Please fill in)

# Install scikit-learn package if necessary:
# pip install -U scikit-learn

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC


def readMatrix(filename: str):
    # Use the code below to read files
    with open(filename, 'r') as fd:
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


def evaluate(output, label) -> float:
    # Use the code below to obtain the accuracy of your algorithm
    error = float((output != label).sum()) * 1. / len(output)
    print('Error: {:2.4f}%'.format(100 * error))

    return error


def main():
    # Load files
    # Note 1: tokenlists (list of all tokens) from MATRIX.TRAIN and MATRIX.TEST are identical
    # Note 2: Spam emails are denoted as class 1, and non-spam ones as class 0.
    dataMatrix_train, tokenlist, category_train = readMatrix('q5_data/MATRIX.TRAIN')
    dataMatrix_test, tokenlist, category_test = readMatrix('q5_data/MATRIX.TEST')

    # Train
    print(dataMatrix_train.shape)
    clf = LinearSVC()
    clf.fit(dataMatrix_train, category_train)
    
    # Test and evluate
    prediction = clf.predict(dataMatrix_test)
    evaluate(prediction, category_test)
    print('\n')

    data_name = ['q5_data/MATRIX.TRAIN.50','q5_data/MATRIX.TRAIN.100','q5_data/MATRIX.TRAIN.200',
    'q5_data/MATRIX.TRAIN.400', 'q5_data/MATRIX.TRAIN.800', 'q5_data/MATRIX.TRAIN.1400']
    errors = np.zeros(6)
    for i in range(6):
        dataMatrix_train, tokenlist, category_train = readMatrix(data_name[i])
        print(f'[{data_name[i]}]:')
        
        clf = LinearSVC()
        clf.fit(dataMatrix_train, category_train)
        prediction = clf.predict(dataMatrix_test)
        decision_function = clf.decision_function(dataMatrix_test)
        support_vector_indices = np.where(np.abs(decision_function) <= 1)[0]
        print(f'numbers of support vectors: {len(support_vector_indices)}')
        
        errors[i] = evaluate(prediction, category_test)
        print('')    

    x_cord = np.array([50,100,200,400,800,1400])
    plt.rcParams.update({'font.size': 22})
    fig = plt.figure(0)
    fig.set_size_inches(18.5, 10.5, forward=True)
    fig.canvas.set_window_title('q5(b)')
    plt.plot(x_cord,100*errors, 'bo-')
    plt.xlabel('training dataset size')
    plt.ylabel('error rate(%)')
    plt.show()

if __name__ == '__main__':
    main()
