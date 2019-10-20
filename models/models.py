from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn import svm

def KNN():
    parameters = [{'n_neighbors': [5, 7, 9, 11]}]
    model = KNeighborsClassifier()
    return parameters, model


def LR(pen, seed):
    parameters = [{'C': [0.001, 0.1, 1, 10, 100]}]
    model = LogisticRegression(penalty=pen, solver='liblinear', random_state=seed)
    return parameters, model


def MLP(seed):
    parameters = [{'hidden_layer_sizes': [(3,), (5,), (7,), (9,)], 'alpha': [0.0001, 0.001]}]
    model = MLPClassifier(activation="logistic", solver="lbfgs", random_state=seed)
    return parameters, model


def RF(seed):
    parameters = [{'criterion': ['entropy', 'gini']}]
    model = RandomForestClassifier(n_estimators=250, random_state=seed)
    return parameters, model


def SVM(seed):
    parameters = [{'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}]
    model = svm.SVC(probability=True, random_state=seed)
    return parameters, model


def VC(seed):
    parameters = [{'l1__C': [0.001, 0.1, 1], 'l2__C': [0.001, 0.1, 1], 'SVM__C': [0.1, 1, 10]}]
    l1 = LogisticRegression(penalty="l1", solver='liblinear', random_state=seed)
    l2 = LogisticRegression(penalty="l2", solver='liblinear', random_state=seed)
    SVM = svm.SVC(probability=True, kernel="rbf", random_state=seed)
    rf = RandomForestClassifier(n_estimators=250, criterion="entropy", random_state=seed)
    vc = VotingClassifier(estimators=[('l1', l1), ('l2', l2), ('SVM', SVM), ('rf', rf)], voting='soft')
    return parameters, vc