import logging
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_svmlight_file
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals.joblib import Memory
import sys

from sklearn.metrics import confusion_matrix

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


mem = Memory("./mycache")

@mem.cache
def load_samples(path_to_svm):

    data = load_svmlight_file(path_to_svm)

    return data[0], data[1]


if __name__ == '__main__':
    X, y = load_samples(sys.argv[1])

    X_test, X_train, y_test, y_train = train_test_split(X, y, test_size= 0.90)

    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)

    y_pred = rfc.predict(X_test)

    y_correct_bool_arr = np.equal(y_test, y_pred)
    y_correct_number = np.count_nonzero(y_correct_bool_arr)
    pred_rate = float(y_correct_number)/y_correct_bool_arr.size

    print(pred_rate)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    print('Confusion matrix, without normalization')
    print(cm)
