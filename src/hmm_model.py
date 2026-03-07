"""
Hidden Markov Model Implementation for Activity Recognition.

This module provides both:
1. Custom HMM implementation with Viterbi and Baum-Welch algorithms
2. hmmlearn library integration

HMM Components:
- Hidden States (Z): Activities (standing, walking, jumping, still)
- Observations (X): Feature vectors from sensor data
- Transition Probabilities (A): P(state_t | state_t-1)
- Emission Probabilities (B): P(observation | state)
- Initial State Probabilities (π): P(state_0)
"""

import numpy as np
from scipy import stats
from scipy.special import logsumexp
from typing import List, Tuple, Optional, Dict
import pickle
import os

from config import ACTIVITIES, N_STATES, MODELS_DIR


class GaussianHMM:
    """
    Gaussian Hidden Markov Model with diagonal covariance.
    
    Implements:
    - Viterbi algorithm for decoding (finding most likely state sequence)
    - Baum-Welch algorithm for training (optimizing parameters)
    - Forward-Backward algorithm for computing state probabilities
    """
    
    def __init__(
        self,
        n_states: int = N_STATES,
        n_features: int = None,
        n_iter: int = 100,
        tol: float = 1e-4,
        random_state: int = 42
    ):
        """
        Initialize Gaussian HMM.
        
        Args:
            n_states: Number of hidden states
            n_features: Number of observation features
            n_iter: Maximum number of EM iterations
            tol: Convergence tolerance
            random_state: Random seed for reproducibility
        """
        self.n_states = n_states
        self.n_features = n_features
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        
        np.random.seed(random_state)
        
        # Model parameters (initialized when fit is called)
        self.pi = None  # Initial state probabilities (n_states,)
        self.A = None   # Transition matrix (n_states, n_states)
        self.means = None  # Emission means (n_states, n_features)
        self.covars = None  # Emission covariances (n_states, n_features) - diagonal
        
        self.is_fitted = False
        self.log_likelihood_history = []
        
    def _init_parameters(self, n_features: int):
        """Initialize model parameters."""
        self.n_features = n_features
        
        # Initial state probabilities - uniform
        self.pi = np.ones(self.n_states) / self.n_states
        
        # Transition matrix - slight bias toward staying in same state
        self.A = np.ones((self.n_states, self.n_states)) * 0.1 / (self.n_states - 1)
        np.fill_diagonal(self.A, 0.9)
        self.A = self.A / self.A.sum(axis=1, keepdims=True)
        
        # Emission parameters - will be initialized from data
        self.means = np.random.randn(self.n_states, n_features) * 0.1
        self.covars = np.ones((self.n_states, n_features)) * 1.0
        
    def _init_from_data(self, X: np.ndarray, lengths: List[int] = None):
        """Initialize parameters from data using k-means-like initialization."""
        n_samples, n_features = X.shape
        self._init_parameters(n_features)
        
        # Use simple k-means initialization for means
        indices = np.random.choice(n_samples, self.n_states, replace=False)
        self.means = X[indices].copy()
        
        # Initialize covariances from data variance
        data_var = np.var(X, axis=0) + 1e-6
        for i in range(self.n_states):
            self.covars[i] = data_var
            
    def _compute_log_emission_probs(self, X: np.ndarray) -> np.ndarray:
        """
        Compute log emission probabilities P(X | state) for all states.
        
        Args:
            X: Observations (n_samples, n_features)
        
        Returns:
            Log probabilities (n_samples, n_states)
        """
        n_samples = X.shape[0]
        log_probs = np.zeros((n_samples, self.n_states))
        
        for state in range(self.n_states):
            # Diagonal covariance - independent Gaussian for each feature
            diff = X - self.means[state]
            log_probs[:, state] = -0.5 * (
                self.n_features * np.log(2 * np.pi) +
                np.sum(np.log(self.covars[state])) +
                np.sum(diff ** 2 / self.covars[state], axis=1)
            )
        
        return log_probs
    
    def _forward(
        self,
        log_emission_probs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward algorithm - compute forward probabilities (alpha).
        
        Args:
            log_emission_probs: Log P(X | state) for each sample and state
        
        Returns:
            log_alpha: Log forward probabilities (n_samples, n_states)
            log_likelihood: Log likelihood of the sequence
        """
        n_samples = log_emission_probs.shape[0]
        log_alpha = np.zeros((n_samples, self.n_states))
        
        # Initialization
        log_alpha[0] = np.log(self.pi + 1e-300) + log_emission_probs[0]
        
        # Recursion
        log_A = np.log(self.A + 1e-300)
        for t in range(1, n_samples):
            for j in range(self.n_states):
                log_alpha[t, j] = logsumexp(
                    log_alpha[t-1] + log_A[:, j]
                ) + log_emission_probs[t, j]
        
        # Termination
        log_likelihood = logsumexp(log_alpha[-1])
        
        return log_alpha, log_likelihood
    
    def _backward(self, log_emission_probs: np.ndarray) -> np.ndarray:
        """
        Backward algorithm - compute backward probabilities (beta).
        
        Args:
            log_emission_probs: Log P(X | state) for each sample and state
        
        Returns:
            log_beta: Log backward probabilities (n_samples, n_states)
        """
        n_samples = log_emission_probs.shape[0]
        log_beta = np.zeros((n_samples, self.n_states))
        
        # Initialization (last time step)
        log_beta[-1] = 0  # log(1) = 0
        
        # Recursion
        log_A = np.log(self.A + 1e-300)
        for t in range(n_samples - 2, -1, -1):
            for i in range(self.n_states):
                log_beta[t, i] = logsumexp(
                    log_A[i] + log_emission_probs[t+1] + log_beta[t+1]
                )
        
        return log_beta
    
    def _compute_posteriors(
        self,
        log_alpha: np.ndarray,
        log_beta: np.ndarray,
        log_emission_probs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute posterior probabilities gamma and xi.
        
        gamma[t, i] = P(state_t = i | observations)
        xi[t, i, j] = P(state_t = i, state_{t+1} = j | observations)
        
        Returns:
            gamma: State posteriors (n_samples, n_states)
            xi: Transition posteriors (n_samples-1, n_states, n_states)
        """
        n_samples = log_alpha.shape[0]
        
        # Gamma
        log_gamma = log_alpha + log_beta
        log_gamma = log_gamma - logsumexp(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma)
        
        # Xi
        log_A = np.log(self.A + 1e-300)
        xi = np.zeros((n_samples - 1, self.n_states, self.n_states))
        
        for t in range(n_samples - 1):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    xi[t, i, j] = (
                        log_alpha[t, i] +
                        log_A[i, j] +
                        log_emission_probs[t+1, j] +
                        log_beta[t+1, j]
                    )
            xi[t] = np.exp(xi[t] - logsumexp(xi[t]))
        
        return gamma, xi
    
    def fit(
        self,
        X: np.ndarray,
        lengths: List[int] = None,
        verbose: bool = True
    ) -> 'GaussianHMM':
        """
        Fit HMM using Baum-Welch (EM) algorithm.
        
        Args:
            X: Observations (n_samples, n_features)
            lengths: Length of each sequence (for multiple sequences)
            verbose: Whether to print progress
        
        Returns:
            self
        """
        if lengths is None:
            lengths = [len(X)]
        
        # Initialize parameters
        self._init_from_data(X, lengths)
        
        prev_log_likelihood = -np.inf
        self.log_likelihood_history = []
        
        for iteration in range(self.n_iter):
            # E-step: compute posteriors
            log_emission_probs = self._compute_log_emission_probs(X)
            log_alpha, log_likelihood = self._forward(log_emission_probs)
            log_beta = self._backward(log_emission_probs)
            gamma, xi = self._compute_posteriors(log_alpha, log_beta, log_emission_probs)
            
            self.log_likelihood_history.append(log_likelihood)
            
            # Check convergence
            if abs(log_likelihood - prev_log_likelihood) < self.tol:
                if verbose:
                    print(f"Converged at iteration {iteration + 1}")
                break
            prev_log_likelihood = log_likelihood
            
            # M-step: update parameters
            self._m_step(X, gamma, xi, lengths)
            
            if verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.n_iter}, "
                      f"Log-likelihood: {log_likelihood:.2f}")
        
        self.is_fitted = True
        return self
    
    def _m_step(
        self,
        X: np.ndarray,
        gamma: np.ndarray,
        xi: np.ndarray,
        lengths: List[int]
    ):
        """M-step: Update model parameters."""
        # Update initial state probabilities
        # Use the first sample of each sequence
        start_indices = np.cumsum([0] + lengths[:-1])
        self.pi = np.mean([gamma[i] for i in start_indices], axis=0)
        self.pi = self.pi / self.pi.sum()  # Normalize
        
        # Update transition matrix
        xi_sum = np.sum(xi, axis=0)
        self.A = xi_sum / (np.sum(xi_sum, axis=1, keepdims=True) + 1e-300)
        
        # Update emission parameters (means and covariances)
        gamma_sum = np.sum(gamma, axis=0)
        
        for i in range(self.n_states):
            # Update mean
            self.means[i] = np.sum(gamma[:, i:i+1] * X, axis=0) / (gamma_sum[i] + 1e-300)
            
            # Update covariance (diagonal)
            diff = X - self.means[i]
            self.covars[i] = np.sum(gamma[:, i:i+1] * (diff ** 2), axis=0) / (gamma_sum[i] + 1e-300)
            self.covars[i] = np.maximum(self.covars[i], 1e-6)  # Prevent zero variance
    
    def viterbi(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Viterbi algorithm - find most likely state sequence.
        
        Args:
            X: Observations (n_samples, n_features)
        
        Returns:
            states: Most likely state sequence (n_samples,)
            log_prob: Log probability of the sequence
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling viterbi()")
        
        n_samples = X.shape[0]
        log_emission_probs = self._compute_log_emission_probs(X)
        
        # Viterbi tables
        viterbi = np.zeros((n_samples, self.n_states))
        backpointer = np.zeros((n_samples, self.n_states), dtype=int)
        
        # Initialization
        log_pi = np.log(self.pi + 1e-300)
        log_A = np.log(self.A + 1e-300)
        viterbi[0] = log_pi + log_emission_probs[0]
        
        # Recursion
        for t in range(1, n_samples):
            for j in range(self.n_states):
                probs = viterbi[t-1] + log_A[:, j]
                backpointer[t, j] = np.argmax(probs)
                viterbi[t, j] = probs[backpointer[t, j]] + log_emission_probs[t, j]
        
        # Termination
        states = np.zeros(n_samples, dtype=int)
        states[-1] = np.argmax(viterbi[-1])
        log_prob = viterbi[-1, states[-1]]
        
        # Backtrack
        for t in range(n_samples - 2, -1, -1):
            states[t] = backpointer[t + 1, states[t + 1]]
        
        return states, log_prob
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the most likely state sequence.
        
        Args:
            X: Observations (n_samples, n_features)
        
        Returns:
            states: Predicted state sequence
        """
        states, _ = self.viterbi(X)
        return states
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute posterior state probabilities.
        
        Args:
            X: Observations (n_samples, n_features)
        
        Returns:
            probs: State probabilities (n_samples, n_states)
        """
        log_emission_probs = self._compute_log_emission_probs(X)
        log_alpha, _ = self._forward(log_emission_probs)
        log_beta = self._backward(log_emission_probs)
        
        log_gamma = log_alpha + log_beta
        log_gamma = log_gamma - logsumexp(log_gamma, axis=1, keepdims=True)
        
        return np.exp(log_gamma)
    
    def score(self, X: np.ndarray) -> float:
        """
        Compute log-likelihood of observations.
        
        Args:
            X: Observations (n_samples, n_features)
        
        Returns:
            Log-likelihood
        """
        log_emission_probs = self._compute_log_emission_probs(X)
        _, log_likelihood = self._forward(log_emission_probs)
        return log_likelihood
    
    def get_transition_matrix(self) -> np.ndarray:
        """Get the learned transition matrix."""
        return self.A.copy() if self.A is not None else None
    
    def get_initial_probabilities(self) -> np.ndarray:
        """Get the learned initial state probabilities."""
        return self.pi.copy() if self.pi is not None else None
    
    def save(self, filepath: str):
        """Save model to file."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'n_states': self.n_states,
                'n_features': self.n_features,
                'pi': self.pi,
                'A': self.A,
                'means': self.means,
                'covars': self.covars,
                'is_fitted': self.is_fitted
            }, f)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.n_states = data['n_states']
        self.n_features = data['n_features']
        self.pi = data['pi']
        self.A = data['A']
        self.means = data['means']
        self.covars = data['covars']
        self.is_fitted = data['is_fitted']
        print(f"Model loaded from {filepath}")


class HMMLearnWrapper:
    """
    Wrapper for hmmlearn library's GaussianHMM.
    Provides the same interface as our custom implementation.
    """
    
    def __init__(
        self,
        n_states: int = N_STATES,
        n_iter: int = 100,
        covariance_type: str = 'diag',
        random_state: int = 42
    ):
        """
        Initialize wrapper for hmmlearn GaussianHMM.
        
        Args:
            n_states: Number of hidden states
            n_iter: Maximum number of EM iterations
            covariance_type: Type of covariance ('diag', 'full', 'spherical')
            random_state: Random seed
        """
        try:
            from hmmlearn.hmm import GaussianHMM
            self.model = GaussianHMM(
                n_components=n_states,
                covariance_type=covariance_type,
                n_iter=n_iter,
                random_state=random_state
            )
            self.using_hmmlearn = True
        except ImportError:
            print("hmmlearn not installed. Using custom implementation.")
            self.model = GaussianHMM(
                n_states=n_states,
                n_iter=n_iter,
                random_state=random_state
            )
            self.using_hmmlearn = False
        
        self.n_states = n_states
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, lengths: List[int] = None, verbose: bool = True) -> 'HMMLearnWrapper':
        """Fit the model."""
        if self.using_hmmlearn:
            self.model.fit(X, lengths)
        else:
            self.model.fit(X, lengths, verbose)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict state sequence."""
        if self.using_hmmlearn:
            return self.model.predict(X)
        else:
            return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get state probabilities."""
        if self.using_hmmlearn:
            return self.model.predict_proba(X)
        else:
            return self.model.predict_proba(X)
    
    def score(self, X: np.ndarray) -> float:
        """Compute log-likelihood."""
        if self.using_hmmlearn:
            return self.model.score(X)
        else:
            return self.model.score(X)
    
    def get_transition_matrix(self) -> np.ndarray:
        """Get transition matrix."""
        if self.using_hmmlearn:
            return self.model.transmat_.copy()
        else:
            return self.model.get_transition_matrix()
    
    def get_initial_probabilities(self) -> np.ndarray:
        """Get initial state probabilities."""
        if self.using_hmmlearn:
            return self.model.startprob_.copy()
        else:
            return self.model.get_initial_probabilities()
    
    def save(self, filepath: str):
        """Save model to file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model from file."""
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        self.is_fitted = True
        print(f"Model loaded from {filepath}")


def create_hmm_model(
    use_hmmlearn: bool = True,
    n_states: int = N_STATES,
    n_iter: int = 100,
    random_state: int = 42
) -> object:
    """
    Factory function to create HMM model.
    
    Args:
        use_hmmlearn: Whether to use hmmlearn library
        n_states: Number of hidden states
        n_iter: Maximum EM iterations
        random_state: Random seed
    
    Returns:
        HMM model instance
    """
    if use_hmmlearn:
        return HMMLearnWrapper(
            n_states=n_states,
            n_iter=n_iter,
            random_state=random_state
        )
    else:
        return GaussianHMM(
            n_states=n_states,
            n_iter=n_iter,
            random_state=random_state
        )


if __name__ == "__main__":
    # Test HMM implementation
    print("=" * 50)
    print("Testing HMM Model Implementation")
    print("=" * 50)
    
    # Generate synthetic test data
    np.random.seed(42)
    n_samples = 200
    n_features = 10
    
    # Create data with 4 distinct clusters (representing activities)
    X = np.zeros((n_samples, n_features))
    true_states = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        state = (i // 50) % 4  # Each activity for 50 samples
        true_states[i] = state
        X[i] = np.random.randn(n_features) * 0.5 + state  # Different mean per state
    
    print(f"\nTest data shape: {X.shape}")
    print(f"True states distribution: {np.bincount(true_states)}")
    
    # Test custom implementation
    print("\n" + "-" * 40)
    print("Testing Custom GaussianHMM")
    print("-" * 40)
    
    hmm = GaussianHMM(n_states=4, n_iter=50)
    hmm.fit(X, verbose=True)
    
    predicted_states = hmm.predict(X)
    print(f"\nPredicted states distribution: {np.bincount(predicted_states)}")
    
    # Check transition matrix
    print("\nLearned Transition Matrix:")
    print(np.round(hmm.get_transition_matrix(), 3))
    
    # Test Viterbi
    states, log_prob = hmm.viterbi(X)
    print(f"\nViterbi log-probability: {log_prob:.2f}")
    
    # Test model save/load
    test_model_path = os.path.join(MODELS_DIR, 'test_hmm.pkl')
    hmm.save(test_model_path)
    
    hmm2 = GaussianHMM(n_states=4)
    hmm2.load(test_model_path)
    print(f"\nModel loaded successfully, is_fitted: {hmm2.is_fitted}")
