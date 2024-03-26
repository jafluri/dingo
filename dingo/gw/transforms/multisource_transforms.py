import numpy as np
import torch
import pandas as pd
from dingo.gw.prior import BBHExtrinsicPriorDict


class SampleExtrinsicMultiSource(object):
    """
    Sample extrinsic parameters and add them to sample in a separate dictionary.
    Here an additional source_dict is provided, which will shift the geocentric time according to a delta_t.
    """

    def __init__(self, extrinsic_prior_dict, source_dict):
        """
        Parameters
        ----------
        extrinsic_prior_dict: dict
            Dictionary of extrinsic parameters and their priors.
        source_dict: dict
            Dictionary of source parameters and their priors.
        """
        self.extrinsic_prior_dict = extrinsic_prior_dict
        self.source_dict = source_dict
        self.prior = BBHExtrinsicPriorDict(extrinsic_prior_dict)
        self.source_prior = BBHExtrinsicPriorDict(source_dict)

    def __call__(self, input_sample):
        """
        Parameters
        ----------
        input_sample: dict
            Dictionary of input samples.
        """
        sample = input_sample.copy()
        extrinsic_parameters = self.prior.sample()
        source_parameters = self.source_prior.sample()
        extrinsic_parameters = {k: float(v) for k, v in extrinsic_parameters.items()}
        extrinsic_parameters["delta_t"] = source_parameters["delta_t"]
        extrinsic_parameters["geocent_time"] += source_parameters["delta_t"]
        sample["extrinsic_parameters"] = extrinsic_parameters
        return sample

    @property
    def reproduction_dict(self):
        return {"extrinsic_prior_dict": self.extrinsic_prior_dict,
                "source_dict": self.source_dict}


class AddNewSource(object):
    """
    This transformation add a new source to the data with independent parameters.
    All parameters are added to the sample dictionars with _{source_name} appended.
    """

    def __init__(self, source_name, source_wfd, source_transform):
        """
        Parameters
        ----------
        source_name: str
            Name of the source to be added.
        source_wfd: WaveformDataset
            WaveformDataset of the source to be added.
        source_transform: Transform
            The transformation to apply to the sample of the source.
        """
        self.source_name = source_name
        self.source_wfd = source_wfd
        self.source_transform = source_transform

    def sample_source_wfd(self):
        """
        Sample a waveform from the source WaveformDataset.
        Returns
        -------
        sample: dict
            Dictionary of the sample.
        """

        index = np.random.randint(0, len(self.source_wfd))
        sample = self.source_wfd.get_no_transform(index)
        return self.source_transform(sample)

    def __call__(self, input_sample):
        """
        Parameters
        ----------
        input_sample: dict
            Dictionary of input samples.
        """

        # Sample the source waveform
        sample = input_sample.copy()
        source_sample = self.sample_source_wfd()

        # combine everything
        for ifo in sample["waveform"].keys():
            sample["waveform"][ifo] += source_sample["waveform"][ifo]
        for param, value in source_sample["parameters"].items():
            sample["parameters"][f"{param}_{self.source_name}"] = value
        # add the delta_t to the parameters
        sample["parameters"][f"delta_t_{self.source_name}"] = source_sample["extrinsic_parameters"]["delta_t"]

        # combine the two samples
        return sample

