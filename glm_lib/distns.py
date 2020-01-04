import numpy as np
import scipy as sp
import scipy.stats
import scipy.special


class ExponentialFamily(object):
    """
    Exponential family distribution.

    [ Note that we employ SciPy distributions heavily. ]

    Parameters
    ----------
    eta: p x m array of natural parameters
    """
    def __init__(self, eta):
        self.eta = eta
        self.dist = None

    def __getattr__(self, name):
        if name in dir(self.dist):
            return getattr(self.dist, name)
        return None

    def gradient(self, x):
        """
        Compute the gradients of negative log-lik ∇-log p(X) with respect to eta.

        Parameters
        ----------
        x: m-length array of observations

        Returns
        -------
        grad: p x m array, gradient for each natural parameter, for each observation
        """
        return self.calc_suff_stats_moments(1) - self.convert_to_suff_stats(x)

    def fisher_info(self):
        """
        Compute the Fisher information I(η) with respect to each of the natural parameters eta.

        Returns
        ----------
        fisher: p x m array, Fisher information for each natural parameter, for each observation.
        """
        return self.calc_suff_stats_moments(2)

    def nll(self, x):
        """
        Compute the negative log-likelihood of observations -log p(X).

        Parameters
        ----------
        x: m-length array of observations

        Returns
        -------
        nll: m-length array of negative log-likelihoods
        """
        raise NotImplementedError

    def calc_suff_stats_moments(self, power):
        """
        Compute moments of sufficient statistics E[T(X)] for current parameterization eta.

        Parameters
        ----------
        power: either 1 or 2, which moment to compute

        Returns
        -------
        moment: p x m array of moments
        """
        raise NotImplementedError

    def convert_to_suff_stats(self, x):
        """
        Extract sufficient statistics T(X) from observations.

        Parameters
        ----------
        x: m-length array of observations

        Returns
        -------
        T: p x m array, sufficient statistics for each parameter, for each observation
        """
        raise NotImplementedError

    @staticmethod
    def n_parameters(x):
        """
        Static method that computes the dimensionality p of eta given observations.

        Parameters
        ----------
        x: m-length array of observations

        Returns
        -------
        p: scalar corresponding to dimensionality of eta
        """
        raise NotImplementedError


class Bernoulli(ExponentialFamily):

    def __init__(self, eta):
        self.dist = sp.stats.bernoulli(p=sp.special.expit(eta.squeeze()))

    def nll(self, x):
        return -self.logpmf(x)

    def convert_to_suff_stats(self, x):
        return x[np.newaxis, :]

    def calc_suff_stats_moments(self, power):
        return self.mean()[np.newaxis, :] if power == 1 else self.var()[np.newaxis, :]

    @staticmethod
    def n_parameters(x):
        return 1


class Poisson(ExponentialFamily):

    def __init__(self, eta):
        self.dist = sp.stats.poisson(mu=np.exp(eta.squeeze()))

    def nll(self, x):
        return -self.logpmf(x)

    def convert_to_suff_stats(self, x):
        return x[np.newaxis, :]

    def calc_suff_stats_moments(self, power):
        return self.mean()[np.newaxis, :] if power == 1 else self.var()[np.newaxis, :]

    @staticmethod
    def n_parameters(x):
        return 1


class Gaussian(ExponentialFamily):

    def __init__(self, eta):
        self.dist = sp.stats.norm(loc=eta.squeeze(), scale=1)

    def nll(self, x):
        return -self.logpdf(x)

    def convert_to_suff_stats(self, x):
        return x[np.newaxis, :]

    def calc_suff_stats_moments(self, power):
        return self.mean()[np.newaxis, :] if power == 1 else self.var()[np.newaxis, :]

    @staticmethod
    def n_parameters(x):
        return 1


class Categorical(ExponentialFamily):

    def __init__(self, eta):
        self.dist = sp.stats.multinomial(n=1, p=sp.special.softmax(eta, axis=1))
        self.n_categories = eta.shape[1]
        self.eye = np.eye(self.n_categories)

    def nll(self, x):
        return -self.logpmf(self.eye[x])

    def convert_to_suff_stats(self, x):
        return self.eye[x].T

    def calc_suff_stats_moments(self, power):
        return self.dist.p.T if power == 1 else (1 / self.dist.p / (1 - self.dist.p)).T

    @staticmethod
    def n_parameters(x):
        return np.max(x) + 1

