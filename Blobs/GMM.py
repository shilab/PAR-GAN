#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 13:22:00 2017
@author: cli
"""

import numpy as np
import numpy.random as npr
from scipy.stats import multivariate_normal
from functools import reduce

class sample_GMM():
    """ Toy dataset containing points sampled from a gaussian mixture distribution.
    The dataset contains 3 sources:
    * samples
    * labels
    * densities
    """
    def __init__(self,
                 num_examples,
                 means=None,
                 variances=None,
                 priors=None,
                 **kwargs):
        rng = kwargs.pop('rng', None)
        if rng is None:
            seed = kwargs.pop('seed', 0)
            rng = np.random.RandomState(seed)

        gaussian_mixture = GMM_distribution(means=means,
                                            variances=variances,
                                            priors=priors,
                                            rng=rng)
        self.means = gaussian_mixture.means
        self.variances = gaussian_mixture.variances
        self.priors = gaussian_mixture.priors

        features, labels = gaussian_mixture.sample(nsamples=num_examples)
        densities = gaussian_mixture.pdf(x=features)

        data = {'samples': features, 'labels': labels, 'density': densities}

        self.data = data
        
        
class GMM_distribution(object):
    """ Gaussian Mixture Distribution
    Parameters
    ----------
    means : tuple of ndarray.
       Specifies the means for the gaussian components.
    variances : tuple of ndarray.
       Specifies the variances for the gaussian components.
    priors : tuple of ndarray
       Specifies the prior distribution of the components.
    """
    def __init__(self,
                 means=None,
                 variances=None,
                 priors=None,
                 rng=None,
                 seed=None):

        if means is None:
            means = map(lambda x: 10.0 * np.array(x),
                        [[0, 0], [1, 1], [-1, -1], [1, -1], [-1, 1]])
        # Number of components
        self.ncomponents = len(means)
        self.dim = means[0].shape[0]
        self.means = means
        # If prior is not specified let prior be flat.
        if priors is None:
            priors = [1.0 / self.ncomponents for _ in range(self.ncomponents)]
        self.priors = priors
        # If variances are not specified let variances be identity
        if variances is None:
            variances = [np.eye(self.dim) for _ in range(self.ncomponents)]
        self.variances = variances

        assert len(means) == len(variances), "Mean variances mismatch"
        assert len(variances) == len(priors), "prior mismatch"

        if rng is None:
            rng = npr.RandomState(seed=seed)
        self.rng = rng

    def _sample_prior(self, nsamples):
        return self.rng.choice(a=self.ncomponents,
                               size=(nsamples, ),
                               replace=True,
                               p=self.priors)

    def sample(self, nsamples):
        # Sampling priors
        samples = []
        fathers = self._sample_prior(nsamples=nsamples).tolist()
        for father in fathers:
            samples.append(
                self._sample_gaussian(self.means[father],
                                      self.variances[father]))
        return np.array(samples), np.array(fathers)

    def _sample_gaussian(self, mean, variance):
        # sampling unit gaussians
        epsilons = self.rng.normal(size=(self.dim, ))

        return mean + np.linalg.cholesky(variance).dot(epsilons)

    def _gaussian_pdf(self, x, mean, variance):
        return multivariate_normal.pdf(x, mean=mean, cov=variance)

    def pdf(self, x):
        "Evaluates the the probability density function at the given point x"
        pdfs = map(lambda m, v, p: p * self._gaussian_pdf(x, m, v), self.means,
                   self.variances, self.priors)
        return reduce(lambda x, y: x + y, pdfs, 0.0)