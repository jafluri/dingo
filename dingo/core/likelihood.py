from multiprocessing import Pool

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


class Likelihood(object):
    def log_likelihood(self, theta):
        raise NotImplementedError("log_likelihood() should be implemented in subclass.")

    def log_likelihood_multi(self, theta: pd.DataFrame, num_processes: int = 1) -> np.ndarray:
        """
        Calculate the log likelihood at multiple points in parameter space. Works with
        multiprocessing.

        This wraps the log_likelihood() method.

        Parameters
        ----------
        theta : dict
            Parameter dictionary. Each value is expected to be a numpy array.
        num_processes : int
            Number of processes to use.

        Returns
        -------
        np.array of log likelihoods
        """
        with threadpool_limits(limits=1, user_api="blas"):
            with Pool(processes=num_processes) as pool:
                # Generator object for theta rows. For idx this yields row idx of theta
                # dataframe, converted to dict, ready to be passed to self.log_likelihood.
                theta_generator = (d[1].to_dict() for d in theta.iterrows())
                # compute log_likelihood with multiprocessing
                log_likelihood = pool.map(self.log_likelihood, theta_generator)

        return np.array(log_likelihood)
