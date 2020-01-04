import numpy as np


class GLM(object):
    """
    Generalized Linear Model.

    Parameters
    ----------
    distn: class of ExponentialFamily
    lr: scalar learning rate
    tol: scalar tolerance
    verbose: boolean
    """
    def __init__(self, distn, lr=0.1, tol=1e-3, verbose=False):
        self.distn = distn
        self.lr = lr
        self.tol = tol
        self.verbose = verbose
        self.theta = None

    def fit(self, x_tr, y_tr):
        """
        Parameters
        ----------
        x_tr: m x n array of training data
        y_tr: m-length array of observations

        Returns
        -------
        nll: scalar average negative log-lik
        """
        m, n = x_tr.shape
        theta = np.zeros((self.distn.n_parameters(y_tr), n))
        itr, prev_nll = 0, float("inf")

        # make initial predictions
        preds = self.distn(x_tr @ theta.T)
        nll = preds.nll(y_tr).mean()

        # fisher scoring steps
        while prev_nll - nll > self.tol:

            if self.verbose:
                print(f"Iter: {itr} NLL: {nll:.2f}")

            prev_nll = nll
            grad = preds.gradient(y_tr) @ x_tr
            fisher_info = preds.fisher_info()
            for i in range(len(theta)):
                x_scaled = fisher_info[i][:, np.newaxis] * x_tr
                grad[i] = np.linalg.inv(x_scaled.T @ x_scaled) @ grad[i]
            theta -= self.lr * grad
            preds = self.distn(x_tr @ theta.T)
            nll = preds.nll(y_tr).mean()
            itr += 1

        self.theta = theta
        return nll

    def predict(self, x_te):
        """
        Parameters
        ----------
        x_te: m x n array of test data.

        Returns
        -------
        preds: m-length distribution objects
        """
        if self.theta is None:
            raise ValueError("Trying to predict before fitting GLM.")
        return self.distn(x_te @ self.theta.T)

