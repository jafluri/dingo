import numpy as np
from scipy.signal import tukey
from scipy.interpolate import interp1d
from bilby.gw.detector import PowerSpectralDensity

from dingo.gw.prior import default_extrinsic_dict
from dingo.gw.prior import BBHExtrinsicPriorDict


def get_window(window_kwargs):
    """Compute window from window_kwargs."""
    type = window_kwargs["type"]
    if type == "tukey":
        roll_off, T, f_s = (
            window_kwargs["roll_off"],
            window_kwargs["T"],
            window_kwargs["f_s"],
        )
        alpha = 2 * roll_off / T
        w = tukey(int(T * f_s), alpha)
        return w
    else:
        raise NotImplementedError(f"Unknown window type {type}.")


def get_window_factor(window):
    """Compute window factor. If window is not provided as array or tensor but as
    window_kwargs, first build the window."""
    if type(window) == dict:
        window = get_window(window)
    return np.sum(window ** 2) / len(window)


def get_extrinsic_prior_dict(extrinsic_prior):
    """Build dict for extrinsic prior by starting with
    default_extrinsic_dict, and overwriting every element for which
    extrinsic_prior is not default.
    TODO: Move to dingo.gw.prior.py?"""
    extrinsic_prior_dict = default_extrinsic_dict.copy()
    for k, v in extrinsic_prior.items():
        if isinstance(v, str) and v.lower() == "default":
            continue
        extrinsic_prior_dict[k] = v
    return extrinsic_prior_dict


def get_mismatch(a, b, domain, asd_file=None):
    """
    Mistmatch is 1 - overlap, where overlap is defined by
    inner(a, b) / sqrt(inner(a, a) * inner(b, b)).
    See e.g. Eq. (44) in https://arxiv.org/pdf/1106.1021.pdf.

    Parameters
    ----------
    a
    b
    domain
    asd_file

    Returns
    -------

    """
    if asd_file is not None:
        # whiten a and b, such that we can use flat-spectrum inner products below
        psd = PowerSpectralDensity(asd_file=asd_file)
        asd_interp = interp1d(
            psd.frequency_array, psd.asd_array, bounds_error=False, fill_value=np.inf
        )
        asd_array = asd_interp(domain.sample_frequencies)
        a /= asd_array
        b /= asd_array
    min_idx = domain.min_idx
    inner_ab = np.sum((a.conj() * b)[min_idx:], axis=0).real
    inner_aa = np.sum((a.conj() * a)[min_idx:], axis=0).real
    inner_bb = np.sum((b.conj() * b)[min_idx:], axis=0).real
    overlap = inner_ab / np.sqrt(inner_aa * inner_bb)
    return 1 - overlap


def get_standardization_dict(
    extrinsic_prior_dict, wfd, selected_parameters, transform=None, multisource_dict=None
):
    """
    Calculates the mean and standard deviation of parameters. This is needed for
    standardizing neural-network input and output.

    Parameters
    ----------
    extrinsic_prior_dict : dict
    wfd : WaveformDataset
    selected_parameters : list[str]
        List of parameters for which to estimate standardization factors.
    transform : Transform
        Operator that will generate samples for parameters contained in
        selected_parameters that are not contained in the intrinsic or extrinsic prior.
        (E.g., H1_time, L1_time_proxy)
    multisource_dict : dict
        If not None, this is a dictionary with keys "source_names" and values
        "source_dict", which are used to construct the extrinsic prior for the
        multisource case.

    Returns
    -------

    """
    # The intrinsic standardization is estimated based on the entire dataset.
    mean_intrinsic, std_intrinsic = wfd.parameter_mean_std()

    # Some of the extrinsic prior parameters have analytic means and standard
    # deviations. If possible, this will either get these, or else it will estimate
    # them numerically.
    ext_prior = BBHExtrinsicPriorDict(extrinsic_prior_dict)
    mean_extrinsic, std_extrinsic = ext_prior.mean_std(ext_prior.keys())

    # Check that overlap between intrinsic and extrinsic parameters is only
    # due to fiducial values (-> std 0)
    for k in std_intrinsic.keys() & std_extrinsic.keys():
        assert std_intrinsic[k] == 0

    # Merge dicts, overwriting fiducial values for parameters (e.g.,
    # luminosity_distance) in intrinsic parameters by the extrinsic ones
    mean = {**mean_intrinsic, **mean_extrinsic}
    std = {**std_intrinsic, **std_extrinsic}

    # add the params for multiple sources
    if multisource_dict is not None:
        for source_name, source_dict in multisource_dict.items():
            source_params = [param for param in selected_parameters if param.endswith(f"_{source_name}")]
            for param in source_params:
                original_param = param.replace(f"_{source_name}", "")

                # this is the only param that the main source does not have
                if original_param == "delta_t":
                    delta_prior = BBHExtrinsicPriorDict(source_dict)
                    delt_m, delta_s = delta_prior.mean_std(["delta_t"])
                    mean[param] = delt_m["delta_t"]
                    std[param] = delta_s["delta_t"]
                    continue

                # if the original param is not already calculated we raise an error
                if original_param not in mean:
                    raise NotImplementedError("All multisource params must be in the intrinsic or extrinsic prior")

                # if its the geocentric time we need to shift it by the mean of delta_t
                delta_mean = 0
                if original_param == "geocent_time":
                    delta_prior = BBHExtrinsicPriorDict(source_dict)
                    delta_mean = delta_prior.mean_std(["delta_t"])[0]["delta_t"]

                mean[param] = mean[original_param] + delta_mean
                std[param] = std[original_param]

    # For all remaining parameters that require standardization, we use the transform
    # to sample these and estimate the mean and standard deviation numerically.
    additional_parameters = [p for p in selected_parameters if p not in mean]

    if additional_parameters:
        num_samples = min(100_000, len(wfd.parameters))
        samples = {p: np.empty(num_samples) for p in additional_parameters}
        for n in range(num_samples):
            sample = {"parameters": wfd.parameters.iloc[n].to_dict()}
            sample = transform(sample)
            for p in additional_parameters:
                # This assumes all of the additional parameters are contained within
                # extrinsic_parameters. We have set it up so this is the case for the
                # GNPE proxies and the detector coalescence times.
                samples[p][n] = sample["extrinsic_parameters"][p]
        mean_additional = {p: np.mean(samples[p]).item() for p in additional_parameters}
        std_additional = {p: np.std(samples[p]).item() for p in additional_parameters}

        mean.update(mean_additional)
        std.update(std_additional)

    standardization_dict = {
        "mean": {k: mean[k] for k in selected_parameters},
        "std": {k: std[k] for k in selected_parameters},
    }
    return standardization_dict
