from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
import numpy as np

class MultiLabelLogisticRegression:
    """
    Implementation of a custom multi-label logistic regression.
    Parameters
    ----------
    strategy : string (default='ovr')
        Multi-label strategy ('simple' | 'ovr')
    n_classes : int (default=40)
        Number of classes
    random_state : int or None (default=0)
        random state for data shuffle
    penalty : string (default='l2')
        Used to specify the norm used in the penalization ('l1' | 'l2' | 'elasticnet' | 'none')
    tolerance : float (default=1e-4)
        Tolerance for stopping criteria
    solver : string (default='lbfgs')
        Algorithm to use in the optimization problem ('newton-cg' | 'lbfgs' | 'liblinear' | 'sag' | 'saga')
    max_iter : int (default=10000)
        Maximum number of iterations taken for the solvers to converge
    multi_class : string (default='auto')
        Logistic regression multi-label scheme ('auto' | 'ovr' | 'multinomial')
    n_jobs : int or None (default=-1)
        Number of used CPU cores
    verbose : int (default=0)
        For the liblinear and lbfgs solvers set verbose to any positive number for verbosity
    """
    def __init__(self, strategy='ovr', n_classes=40, random_state=0,
                penalty='l2', tolerance=1e-4, solver='lbfgs',
                max_iter=10000, multi_class='auto', n_jobs=-1, verbose=0):
        assert strategy == 'simple' or strategy == 'ovr', 'Multi-label strategy is invalid!'
        lr_classifier = LogisticRegression(penalty=penalty, tol=tolerance, random_state=random_state,
                                            solver=solver, max_iter=max_iter, multi_class=multi_class,
                                            verbose=verbose, n_jobs=n_jobs)
        if strategy == 'simple':
            self.ml_classifier = MultiOutputClassifier(estimator=lr_classifier, n_jobs=n_jobs)
        elif strategy == 'ovr':
            self.ml_classifier = OneVsRestClassifier(estimator=lr_classifier, n_jobs=n_jobs)

    def fit(self, X, y):
        self.ml_classifier = self.ml_classifier.fit(X, y)

    def predict(self, X):
        return self.ml_classifier.predict(X)

    def get_coefficients(self):
        coef_list = [estimator.coef_ for estimator in self.ml_classifier.estimators_]
        return np.concatenate(coef_list).transpose()
