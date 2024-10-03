from multiprocessing import Pool

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


class Likelihood(object):
    def log_likelihood(self, theta):
        raise NotImplementedError("log_likelihood() should be implemented in subclass.")

    def log_likelihood_multi(
        self, theta: pd.DataFrame | None, num_processes: int = 1, theta_generator = None,
    ) -> np.ndarray:
        """
        Calculate the log likelihood at multiple points in parameter space. Works with
        multiprocessing.

        This wraps the log_likelihood() method.

        Parameters
        ----------
        theta : pd.DataFrame
            Parameters values at which to evaluate likelihood.
        num_processes : int
            Number of processes to use.
        theta_generator : generator
            Instead of passing theta, pass a generator

        Returns
        -------
        np.array of log likelihoods
        """
        with threadpool_limits(limits=1, user_api="blas"):

            # Generator object for theta rows. For idx this yields row idx of
            # theta dataframe, converted to dict, ready to be passed to
            # self.log_likelihood.
            if theta is not None and theta_generator is not None:
                raise ValueError("Only one of theta or theta_generator should be provided.")
            elif theta is not None and theta_generator is None:
                theta_generator = (d[1].to_dict() for d in theta.iterrows())
            elif theta_generator is None:
                raise ValueError("Either theta or theta_generator must be provided.")

            if num_processes > 1:
                with Pool(processes=num_processes) as pool:
                    log_likelihood = pool.map(self.log_likelihood, theta_generator)
            else:
                log_likelihood = list(map(self.log_likelihood, theta_generator))

        return np.array(log_likelihood)
