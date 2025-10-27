import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans

import pickle

def svm_test(X, y, repeat=10):
    random_states = [182318 + i for i in range(repeat)]
    macro_f1_list = []
    micro_f1_list = []
    for i in range(repeat):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=random_states[i])
        svm = LinearSVC(dual=False)
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        micro_f1 = f1_score(y_test, y_pred, average='micro')
        macro_f1_list.append(macro_f1)
        micro_f1_list.append(micro_f1)

    return np.mean(macro_f1_list), np.mean(micro_f1_list)

