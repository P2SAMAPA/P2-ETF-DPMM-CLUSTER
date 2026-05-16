import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.preprocessing import StandardScaler

class DPMMVariational:
    def __init__(self, n_components=10, alpha=1.0, max_iter=100, tol=1e-3):
        self.n_components = n_components
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.vb_params = None
        self.elbo_ = []

    def _initialize(self, X):
        n, d = X.shape
        # Stick‑breaking variational parameters
        self.gamma = np.ones(self.n_components) * self.alpha / self.n_components  # Dirichlet prior for sticks
        # Gaussian component parameters (mean, precision)
        self.m = np.random.randn(self.n_components, d) * 0.1
        self.beta = np.ones(self.n_components)  # scale for precision (Wishart prior)
        self.W = np.array([np.eye(d) for _ in range(self.n_components)])  # precision matrix (Wishart scale)
        self.nu = d * np.ones(self.n_components)  # degrees of freedom
        # Responsibilities
        self.r = np.random.rand(n, self.n_components)
        self.r /= self.r.sum(axis=1, keepdims=True)

    def _e_step(self, X):
        n, d = X.shape
        # Compute expected log likelihood under current parameters
        log_rho = np.zeros((n, self.n_components))
        for k in range(self.n_components):
            # Expected precision matrix: nu_k * W_k
            prec = self.nu[k] * self.W[k]
            # Log determinant
            _, logdet = np.linalg.slogdet(prec)
            # Quadratic term
            diff = X - self.m[k]
            quad = np.einsum('ij,ji->i', diff @ prec, diff.T)  # diagonal
            log_rho[:, k] = 0.5 * (logdet - d * np.log(2*np.pi) - quad)
        # Add expected stick weights (using digamma)
        # For stick-breaking: E[log V_k] = digamma(gamma[k]) - digamma(gamma[k]+gamma[k+1])
        # and E[log(1-V_k)] = digamma(gamma[k+1]) - digamma(gamma[k]+gamma[k+1])
        cum_gamma = np.cumsum(self.gamma)
        log_v = np.zeros(self.n_components)
        log_one_minus_v = np.zeros(self.n_components)
        for k in range(self.n_components):
            if k < self.n_components - 1:
                from scipy.special import digamma
                log_v[k] = digamma(self.gamma[k]) - digamma(self.gamma[k] + self.gamma[k+1])
                log_one_minus_v[k] = digamma(self.gamma[k+1]) - digamma(self.gamma[k] + self.gamma[k+1])
            else:
                log_v[k] = digamma(self.gamma[k]) - digamma(self.gamma[k] + 1e-10)
                log_one_minus_v[k] = 0.0
        # Expected log of component weight: log(π_k) = sum_{j<k} log(1-V_j) + log(V_k)
        log_pi = np.zeros(self.n_components)
        cum = 0.0
        for k in range(self.n_components):
            log_pi[k] = cum + log_v[k]
            cum += log_one_minus_v[k]
        log_rho += log_pi
        # Normalise
        log_rho_max = log_rho.max(axis=1, keepdims=True)
        rho = np.exp(log_rho - log_rho_max)
        r = rho / rho.sum(axis=1, keepdims=True)
        return r

    def _m_step(self, X, r):
        n, d = X.shape
        Nk = r.sum(axis=0)
        # Update m (mean)
        self.m = (r.T @ X) / Nk[:, None]
        # Update beta (scale for precision) – simplified
        self.beta = 1.0  # we keep fixed for simplicity
        # Update W (precision matrix) – using empirical covariance
        for k in range(self.n_components):
            if Nk[k] > 1e-6:
                diff = X - self.m[k]
                weighted_cov = (r[:, k:k+1] * diff).T @ diff
                self.W[k] = weighted_cov / Nk[k]
                # Regularise to avoid singular
                self.W[k] += 1e-6 * np.eye(d)
        # Update gamma (stick‑breaking) using expected values
        from scipy.special import digamma
        # E[z_k] = Nk (responsibility sum)
        # For stick-breaking, gamma_k = alpha_k + Nk, where alpha_k = 1 for all?
        self.gamma = 1.0 + Nk   # prior = 1
        # Update nu (degrees of freedom) – simple heuristic
        self.nu = d * np.ones(self.n_components)

    def fit(self, X):
        n, d = X.shape
        self._initialize(X)
        prev_elbo = -np.inf
        for it in range(self.max_iter):
            r = self._e_step(X)
            self._m_step(X, r)
            # Compute ELBO (simplified)
            elbo = 0.0
            # Likelihood term
            for k in range(self.n_components):
                prec = self.nu[k] * self.W[k]
                diff = X - self.m[k]
                quad = np.einsum('ij,ji->i', diff @ prec, diff.T)
                log_lik = 0.5 * (np.linalg.slogdet(prec)[1] - d*np.log(2*np.pi) - quad)
                elbo += (r[:,k] * log_lik).sum()
            # KL term (simplified approximation)
            # Not accurate but enough for convergence detection
            if it > 0 and abs(elbo - prev_elbo) < self.tol:
                break
            prev_elbo = elbo
            self.elbo_.append(elbo)
        return self

    def predict_proba(self, X):
        r = self._e_step(X)
        return r

    def get_cluster_means(self):
        return self.m

    def get_cluster_weights(self):
        cum_gamma = np.cumsum(self.gamma)
        from scipy.special import digamma
        log_v = np.zeros(self.n_components)
        for k in range(self.n_components):
            if k < self.n_components - 1:
                log_v[k] = digamma(self.gamma[k]) - digamma(self.gamma[k] + self.gamma[k+1])
            else:
                log_v[k] = digamma(self.gamma[k]) - digamma(self.gamma[k] + 1e-10)
        log_pi = np.zeros(self.n_components)
        cum = 0.0
        for k in range(self.n_components):
            log_pi[k] = cum + log_v[k]
            cum += digamma(self.gamma[k+1]) - digamma(self.gamma[k] + self.gamma[k+1]) if k < self.n_components-1 else 0.0
        weights = np.exp(log_pi)
        return weights / weights.sum()
