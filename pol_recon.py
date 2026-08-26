import copy
import re
import warnings
import numpy as np
from scipy.interpolate import PchipInterpolator

from enterprise.signals import utils
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
import matplotlib as mpl

# --- global style (adjust once, reuse everywhere) ---
mpl.rcParams.update({
    "figure.dpi": 400,
    "savefig.dpi": 400,
    "font.size": 18,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 14,
    "axes.linewidth": 1.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
})



# ============================================================
# Utilities
# ============================================================

_POL_PROCESS_RE = re.compile(
    r"^(?P<psr>.+)_pol_cal_(?P<axis>[xyzXYZ])_(?P<band>.+)$"
)


def _parse_pol_process_name(process_name):
    """
    Parse e.g.

        J1022+1001_pol_cal_x_40CM

    into

        psr  = J1022+1001
        axis = X
        band = 40CM
    """

    match = _POL_PROCESS_RE.match(process_name)

    if match is None:
        raise ValueError(
            f"{process_name!r} is not recognised as a polarization GP.\n"
            "Expected something like:\n"
            "    J1022+1001_pol_cal_x_40CM"
        )

    info = match.groupdict()

    return (
        info["psr"],
        info["axis"].upper(),
        info["band"],
    )


def _find_signal(pta, process_name):
    """
    Find an Enterprise signal by its full name.
    """

    for model in pta.pulsarmodels:
        for signal in model._signals:
            if signal.name == process_name:
                return signal

    available = sorted(
        signal.name
        for model in pta.pulsarmodels
        for signal in model._signals
        if "pol_cal_" in signal.name
    )

    raise KeyError(
        f"Could not find {process_name!r}.\n\n"
        "Available polarization processes:\n"
        + "\n".join(available)
    )


def _get_signal_psr(signal):
    """
    Recover the Pulsar object attached to the basis function.

    Your custom BasisCommonGP constructs the function using

        basisFunction(..., psr=psr)

    so Enterprise normally retains it on the Function instance.
    """

    candidates = [
        getattr(signal, "_psr", None),
        getattr(getattr(signal, "_bases", None), "_psr", None),
        getattr(getattr(signal, "_bases", None), "psr", None),
    ]

    for candidate in candidates:
        if (
            candidate is not None
            and hasattr(candidate, "toas")
            and hasattr(candidate, "distort_vect")
        ):
            return candidate

    raise RuntimeError(
        f"Could not recover the Pulsar object for {signal.name!r}.\n"
        "Expected signal._bases to retain the psr supplied when "
        "your custom basis was constructed."
    )


def _normalise_flags(values):
    """
    Convert Enterprise flag arrays to ordinary strings.
    """

    values = np.asarray(values)

    return np.asarray([
        (
            value.decode()
            if isinstance(value, (bytes, np.bytes_))
            else str(value)
        )
        for value in values
    ])


def _select_posterior_samples(
    posterior_samples,
    N,
    selection,
    seed,
):
    """
    Support either

        ndarray[nposterior, ndim]

    or a poco-style dictionary containing "samples".
    """

    logpost = None

    if isinstance(posterior_samples, dict):

        samples = np.asarray(
            posterior_samples["samples"],
            dtype=float,
        )

        if (
            "log_likelihood" in posterior_samples
            and "log_prior" in posterior_samples
        ):
            logpost = (
                np.asarray(
                    posterior_samples["log_likelihood"],
                    dtype=float,
                )
                +
                np.asarray(
                    posterior_samples["log_prior"],
                    dtype=float,
                )
            )

    else:
        samples = np.asarray(
            posterior_samples,
            dtype=float,
        )

    if samples.ndim != 2:
        raise ValueError(
            "posterior_samples must have shape "
            "(nposterior, ndim)."
        )

    N = int(N)

    if N <= 0:
        raise ValueError("N must be positive.")

    if logpost is not None:
        valid = np.flatnonzero(
            np.isfinite(logpost)
        )
    else:
        valid = np.arange(
            samples.shape[0]
        )

    if valid.size == 0:
        raise ValueError(
            "No usable posterior samples."
        )

    N = min(
        N,
        valid.size,
    )

    if selection == "random":

        rng = np.random.default_rng(
            seed
        )

        indices = rng.choice(
            valid,
            size=N,
            replace=False,
        )

    elif selection == "highest":

        if logpost is None:
            raise ValueError(
                "selection='highest' requires "
                "log_likelihood and log_prior."
            )

        order = np.argsort(
            logpost[valid]
        )

        indices = valid[
            order[-N:]
        ][::-1]

    elif selection == "first":

        indices = valid[:N]

    else:
        raise ValueError(
            "selection must be "
            "'random', 'highest', or 'first'."
        )

    return samples, indices


def _collapse_duplicate_times(
    mjd,
    values,
):
    """
    PCHIP requires unique x values.

    If multiple TOAs have exactly the same MJD, average their
    distortion values.
    """

    mjd = np.asarray(
        mjd,
        dtype=float,
    )

    values = np.asarray(
        values,
        dtype=float,
    )

    good = (
        np.isfinite(mjd)
        & np.isfinite(values)
    )

    mjd = mjd[good]
    values = values[good]

    order = np.argsort(
        mjd
    )

    mjd = mjd[order]
    values = values[order]

    unique_mjd, inverse = np.unique(
        mjd,
        return_inverse=True,
    )

    sums = np.zeros(
        unique_mjd.size,
        dtype=float,
    )

    counts = np.zeros(
        unique_mjd.size,
        dtype=float,
    )

    np.add.at(
        sums,
        inverse,
        values,
    )

    np.add.at(
        counts,
        inverse,
        1.0,
    )

    unique_values = (
        sums / counts
    )

    return (
        unique_mjd,
        unique_values,
    )


def _interpolate_distortion(
    observed_mjd,
    observed_distortion,
    grid_mjd,
    max_gap_days=None,
):
    """
    Interpolate d(t), NOT the GP.

    This creates the continuous approximation

        d(t*) F(t*) a

    from the distortion vector defined at observed TOAs.
    """

    t, d = _collapse_duplicate_times(
        observed_mjd,
        observed_distortion,
    )

    if t.size < 2:
        raise ValueError(
            "At least two unique TOAs are required "
            "to interpolate the distortion vector."
        )

    interpolator = PchipInterpolator(
        t,
        d,
        extrapolate=False,
    )

    d_grid = interpolator(
        grid_mjd
    )

    # Optional: do not pretend the distortion is constrained
    # across very large gaps in the data.
    if max_gap_days is not None:

        max_gap_days = float(
            max_gap_days
        )

        for left, right in zip(
            t[:-1],
            t[1:],
        ):
            if (
                right - left
                > max_gap_days
            ):
                mask = (
                    (grid_mjd > left)
                    & (grid_mjd < right)
                )

                d_grid[mask] = np.nan

    return d_grid


def _get_coefficient_vector(
    coefficient_dict,
    process_name,
):
    """
    Extract conditional coefficients robustly.
    """

    exact_key = (
        process_name
        + "_coefficients"
    )

    if exact_key in coefficient_dict:
        return np.asarray(
            coefficient_dict[exact_key],
            dtype=float,
        )

    candidates = [
        key
        for key in coefficient_dict
        if (
            process_name in key
            and "coeff" in key.lower()
        )
    ]

    if len(candidates) == 1:
        return np.asarray(
            coefficient_dict[candidates[0]],
            dtype=float,
        )

    raise KeyError(
        f"Could not find coefficients for {process_name!r}.\n\n"
        "Available coefficient keys:\n"
        + "\n".join(
            sorted(
                str(key)
                for key in coefficient_dict
                if "coeff" in str(key).lower()
            )
        )
    )


# ============================================================
# Main reconstruction
# ============================================================

def reconstruct_smooth_pol_gp(
    pta,
    posterior_samples,
    process_name,
    N=100,
    n_grid=5000,
    selection="random",
    seed=1234,
    flagname="B",
    coefficient_mode="mean",
    phiinv_method="cliques",
    max_gap_days=None,
    validate=True,
):
    """
    Reconstruct N smooth, distortion-projected polarization GPs.

    Parameters
    ----------
    pta
        Enterprise PTA object.

    posterior_samples
        Either:

            ndarray, shape (nposterior, ndim)

        or a dictionary such as poco_result containing:

            ["samples"]
            ["log_likelihood"]   optional
            ["log_prior"]        optional

    process_name : str
        Example:

            "J1022+1001_pol_cal_x_40CM"

    N : int
        Number of posterior reconstructions.

    n_grid : int
        Number of points in the dense temporal grid.

    selection : {"random", "highest", "first"}

    seed : int

    flagname : str
        Flag used by create_polarization_signals().
        Your current default is "B".

    coefficient_mode : {"mean", "sample"}

        "mean":
            Conditional GP mean coefficients for each posterior
            hyperparameter sample.

        "sample":
            One conditional coefficient realization for each
            posterior hyperparameter sample.

    phiinv_method : str

    max_gap_days : float or None
        If supplied, dense curves are set to NaN inside observing
        gaps larger than this number of days.

    validate : bool
        Verify that our reconstructed distorted Fourier basis
        reproduces the original Enterprise basis at the TOAs.

    Returns
    -------
    result : dict

        result["mjd"]
            Dense MJD grid, shape (n_grid,)

        result["gp_us"]
            THE MAIN OUTPUT:
            distortion-projected smooth GP reconstructions,
            shape (N, n_grid)

        result["gp_seconds"]
            Same in seconds.

        result["boost_process"]
            Undistorted polarization boost/distortion process b_axis(t).
            This quantity is dimensionless. It must NOT be converted to
            seconds or microseconds.

        result["distortion_grid"]
            Interpolated distortion d(t*).

        result["observed_gp_us"]
            Exact process at the original TOAs, shape
            (N, n_original_toas).

        result["frequencies"]
            Fitted Fourier frequencies.

        result["phases"]
            Recovered Fourier phases.

        result["selected_indices"]
            Posterior samples used.

        result["band_mask"]
            Original TOAs belonging to this process/band.
    """

    # ---------------------------------------------------------
    # Decode process
    # ---------------------------------------------------------

    psr_name, axis, band = (
        _parse_pol_process_name(
            process_name
        )
    )

    axis_index = {
        "X": 0,
        "Y": 1,
        "Z": 2,
    }[axis]

    # ---------------------------------------------------------
    # Find Enterprise signal and Pulsar
    # ---------------------------------------------------------

    signal = _find_signal(
        pta,
        process_name,
    )

    psr = _get_signal_psr(
        signal
    )

    if psr.name != psr_name:
        raise RuntimeError(
            f"Process name refers to {psr_name}, "
            f"but attached Pulsar is {psr.name}."
        )

    original_toas = np.asarray(
        psr.toas,
        dtype=float,
    )

    original_mjd = (
        original_toas
        / 86400.0
    )

    distortion = np.asarray(
        psr.distort_vect,
        dtype=float,
    )

    if distortion.ndim != 2:
        raise ValueError(
            "psr.distort_vect must be a 2D array."
        )

    if distortion.shape[0] != original_toas.size:
        raise ValueError(
            "distort_vect and TOA arrays have "
            "different lengths."
        )

    if distortion.shape[1] < 3:
        raise ValueError(
            "distort_vect must contain X, Y, Z columns."
        )

    distort_axis = distortion[
        :,
        axis_index,
    ]

    # ---------------------------------------------------------
    # Band mask
    # ---------------------------------------------------------

    if not hasattr(psr, "flags"):
        raise AttributeError(
            "Pulsar object has no flags attribute."
        )

    if flagname not in psr.flags:
        raise KeyError(
            f"Flag {flagname!r} not found.\n"
            f"Available flags: {list(psr.flags.keys())}"
        )

    flag_values = _normalise_flags(
        psr.flags[flagname]
    )

    band_mask = (
        flag_values
        == str(band)
    )

    if not np.any(
        band_mask
    ):
        raise ValueError(
            f"No TOAs satisfy "
            f"{flagname}={band!r}."
        )

    band_mjd = original_mjd[
        band_mask
    ]

    band_distortion = distort_axis[
        band_mask
    ]

    # ---------------------------------------------------------
    # Select posterior samples
    # ---------------------------------------------------------

    samples, selected_indices = (
        _select_posterior_samples(
            posterior_samples,
            N=N,
            selection=selection,
            seed=seed,
        )
    )

    # ---------------------------------------------------------
    # Build original Enterprise basis once
    # ---------------------------------------------------------

    first_params = pta.map_params(
        samples[
            selected_indices[0]
        ]
    )

    F_enterprise = np.asarray(
        signal.get_basis(
            params=first_params
        ),
        dtype=float,
    )

    labels = np.asarray(
        signal._labels,
        dtype=float,
    ).ravel()

    if (
        F_enterprise.shape[0]
        != original_toas.size
    ):
        raise RuntimeError(
            "Enterprise basis row count does not "
            "match pulsar TOAs."
        )

    if (
        labels.size
        != F_enterprise.shape[1]
    ):
        raise RuntimeError(
            "Enterprise Fourier labels do not "
            "match basis columns."
        )

    if labels.size % 2:
        raise RuntimeError(
            "Expected sine/cosine Fourier pairs."
        )

    # Your basis explicitly uses:
    #
    #     Ffreqs = np.repeat(f, 2)
    #
    if not np.allclose(
        labels[::2],
        labels[1::2],
    ):
        raise RuntimeError(
            "Expected labels arranged as "
            "[f1,f1,f2,f2,...]."
        )

    frequencies = (
        labels[::2].copy()
    )

    n_modes = (
        frequencies.size
    )

    # ---------------------------------------------------------
    # Recover any Fourier phase used by the actual fitted basis.
    #
    # Enterprise matrix:
    #
    #   d_i sin(wt + phi)
    #   d_i cos(wt + phi)
    #
    # Divide out d_i and infer phi.
    # ---------------------------------------------------------

    phases = np.zeros(
        n_modes,
        dtype=float,
    )

    distortion_tolerance = 1e-14

    for mode_index, freq in enumerate(
        frequencies
    ):

        fsin = F_enterprise[
            :,
            2 * mode_index,
        ]

        fcos = F_enterprise[
            :,
            2 * mode_index + 1,
        ]

        usable = (
            band_mask
            & np.isfinite(distort_axis)
            & (
                np.abs(distort_axis)
                > distortion_tolerance
            )
        )

        indices = np.flatnonzero(
            usable
        )

        if indices.size == 0:
            raise RuntimeError(
                f"No usable distortion values for "
                f"Fourier mode {mode_index}."
            )

        sin_component = (
            fsin[indices]
            / distort_axis[indices]
        )

        cos_component = (
            fcos[indices]
            / distort_axis[indices]
        )

        measured_angle = np.arctan2(
            sin_component,
            cos_component,
        )

        phase_samples = (
            measured_angle
            - 2.0
            * np.pi
            * original_toas[indices]
            * freq
        )

        # Circular mean
        phases[mode_index] = np.angle(
            np.mean(
                np.exp(
                    1j * phase_samples
                )
            )
        )

    # ---------------------------------------------------------
    # Reconstruct latent Fourier basis at original TOAs.
    # ---------------------------------------------------------

    observed_angle = (
        2.0
        * np.pi
        * original_toas[:, None]
        * frequencies[None, :]
        + phases[None, :]
    )

    F_latent_observed = np.empty(
        (
            original_toas.size,
            2 * n_modes,
        ),
        dtype=float,
    )

    F_latent_observed[
        :,
        ::2
    ] = np.sin(
        observed_angle
    )

    F_latent_observed[
        :,
        1::2
    ] = np.cos(
        observed_angle
    )

    # ---------------------------------------------------------
    # Recreate your ACTUAL fitted basis:
    #
    #     mask * d_axis * F_fourier
    # ---------------------------------------------------------

    F_reconstructed = (
        F_latent_observed
        * distort_axis[:, None]
        * band_mask[:, None]
    )

    basis_max_error = float(
        np.nanmax(
            np.abs(
                F_reconstructed
                - F_enterprise
            )
        )
    )

    if validate:

        if not np.allclose(
            F_reconstructed,
            F_enterprise,
            rtol=1e-7,
            atol=1e-12,
        ):
            raise RuntimeError(
                "Dense reconstruction setup does not reproduce "
                "the original Enterprise basis.\n"
                f"Maximum basis error = {basis_max_error:.6g}\n\n"
                "Do not trust the reconstruction until this "
                "validation passes."
            )

    # ---------------------------------------------------------
    # Dense time grid
    # ---------------------------------------------------------

    grid_mjd = np.linspace(
        band_mjd.min(),
        band_mjd.max(),
        int(n_grid),
    )

    grid_toas = (
        grid_mjd
        * 86400.0
    )

    # ---------------------------------------------------------
    # Continuous distortion d(t*)
    #
    # This is the ONLY interpolation step.
    # ---------------------------------------------------------

    distortion_grid = (
        _interpolate_distortion(
            observed_mjd=band_mjd,
            observed_distortion=band_distortion,
            grid_mjd=grid_mjd,
            max_gap_days=max_gap_days,
        )
    )

    # ---------------------------------------------------------
    # Exact Fourier basis on dense grid
    # ---------------------------------------------------------

    grid_angle = (
        2.0
        * np.pi
        * grid_toas[:, None]
        * frequencies[None, :]
        + phases[None, :]
    )

    F_latent_grid = np.empty(
        (
            grid_mjd.size,
            2 * n_modes,
        ),
        dtype=float,
    )

    F_latent_grid[
        :,
        ::2
    ] = np.sin(
        grid_angle
    )

    F_latent_grid[
        :,
        1::2
    ] = np.cos(
        grid_angle
    )

    # The actual dense polarization timing-delay basis.
    F_projected_grid = (
        distortion_grid[:, None]
        * F_latent_grid
    )

    # ---------------------------------------------------------
    # Conditional coefficient reconstruction
    # ---------------------------------------------------------

    conditional_gp = utils.ConditionalGP(
        pta,
        phiinv_method=phiinv_method,
    )

    projected_reconstructions = []
    latent_reconstructions = []
    observed_reconstructions = []
    coefficient_samples = []

    for posterior_index in selected_indices:

        theta = samples[
            posterior_index
        ]

        params = pta.map_params(
            theta
        )

        if coefficient_mode == "mean":

            coefficient_dict = (
                conditional_gp
                .get_mean_coefficients(
                    params
                )
            )

        elif coefficient_mode == "sample":

            sampled = (
                conditional_gp
                .sample_coefficients(
                    params,
                    n=1,
                )
            )

            if isinstance(
                sampled,
                dict,
            ):
                coefficient_dict = sampled
            else:
                coefficient_dict = sampled[0]

        else:
            raise ValueError(
                "coefficient_mode must be "
                "'mean' or 'sample'."
            )

        coeff = _get_coefficient_vector(
            coefficient_dict,
            process_name,
        )

        if (
            coeff.size
            != F_latent_grid.shape[1]
        ):
            raise RuntimeError(
                f"ConditionalGP returned {coeff.size} "
                f"coefficients, but the Fourier basis has "
                f"{F_latent_grid.shape[1]} columns."
            )

        # ---------------------------------------------
        # Large underlying Fourier field
        # ---------------------------------------------

        latent_gp = (
            F_latent_grid
            @ coeff
        )

        # ---------------------------------------------
        # Quantity that enters timing residuals:
        #
        #     d(t*) * F(t*) @ a
        # ---------------------------------------------

        projected_gp = (
            F_projected_grid
            @ coeff
        )

        # ---------------------------------------------
        # Exact original-TOA process
        # ---------------------------------------------

        observed_gp = (
            F_reconstructed
            @ coeff
        )

        projected_reconstructions.append(
            projected_gp
        )

        latent_reconstructions.append(
            latent_gp
        )

        observed_reconstructions.append(
            observed_gp
        )

        coefficient_samples.append(
            coeff
        )

    projected_reconstructions = np.asarray(
        projected_reconstructions
    )

    latent_reconstructions = np.asarray(
        latent_reconstructions
    )

    observed_reconstructions = np.asarray(
        observed_reconstructions
    )

    coefficient_samples = np.asarray(
        coefficient_samples
    )

    # ---------------------------------------------------------
    # Posterior summaries
    # ---------------------------------------------------------

    lower_us, median_us, upper_us = (
        np.nanquantile(
            projected_reconstructions
            * 1e6,
            [0.05, 0.50, 0.95],
            axis=0,
        )
    )

    # ---------------------------------------------------------
    # Useful diagnostics
    # ---------------------------------------------------------

    median_abs_distortion = float(
        np.nanmedian(
            np.abs(
                band_distortion
            )
        )
    )

    boost_peak = float(
        np.nanmax(
            np.abs(
                latent_reconstructions
            )
        )
    )

    projected_peak_us = float(
        np.nanmax(
            np.abs(
                projected_reconstructions
                * 1e6
            )
        )
    )

    return {
        # --------------------------------------------
        # Main requested output
        # --------------------------------------------
        "mjd": grid_mjd,

        "gp_seconds":
            projected_reconstructions,

        "gp_us":
            projected_reconstructions * 1e6,

        "lower_us":
            lower_us,

        "median_us":
            median_us,

        "upper_us":
            upper_us,

        # --------------------------------------------
        # Underlying boost / instrumental-distortion field
        #
        # delta_t_axis(t) = Delta_axis(t) * b_axis(t)
        #
        # Delta_axis has units of seconds, therefore b_axis(t)
        # is dimensionless. Do NOT multiply this by 1e6.
        # --------------------------------------------
        "boost_process":
            latent_reconstructions,

        # Short physics notation alias.
        "b_process":
            latent_reconstructions,

        # Backwards-compatible neutral alias (dimensionless).
        "latent_gp":
            latent_reconstructions,

        # --------------------------------------------
        # Distortion
        # --------------------------------------------
        "distortion_grid":
            distortion_grid,

        "observed_distortion":
            distort_axis,

        # --------------------------------------------
        # Original TOA reconstruction
        # --------------------------------------------
        "observed_mjd":
            original_mjd,

        "observed_gp_seconds":
            observed_reconstructions,

        "observed_gp_us":
            observed_reconstructions * 1e6,

        "band_mask":
            band_mask,

        # --------------------------------------------
        # Fourier information
        # --------------------------------------------
        "frequencies":
            frequencies,

        "phases":
            phases,

        "coefficients":
            coefficient_samples,

        # --------------------------------------------
        # Metadata
        # --------------------------------------------
        "process_name":
            process_name,

        "psr_name":
            psr_name,

        "axis":
            axis,

        "band":
            band,

        "selected_indices":
            selected_indices,

        "coefficient_mode":
            coefficient_mode,

        # --------------------------------------------
        # Validation / diagnostics
        # --------------------------------------------
        "basis_max_error":
            basis_max_error,

        "median_abs_distortion":
            median_abs_distortion,

        "boost_peak":
            boost_peak,

        "latent_peak":
            boost_peak,

        "projected_peak_us":
            projected_peak_us,
    }



def reconstruct_pol_xyz_total(
    pta,
    posterior_samples,
    psr_name,
    band,
    *,
    N=100,
    n_grid=5000,
    selection="random",
    seed=1234,
    flagname="B",
    coefficient_mode="mean",
    phiinv_method="cliques",
    max_gap_days=None,
    validate=True,
    credible_level=0.90,
):
    """
    Reconstruct X, Y, Z polarization GPs and their total process
    for one pulsar and one observing band.

    The function calls reconstruct_smooth_pol_gp() for:

        {psr_name}_pol_cal_x_{band}
        {psr_name}_pol_cal_y_{band}
        {psr_name}_pol_cal_z_{band}

    and computes

        total = X + Y + Z

    sample-by-sample.

    Parameters
    ----------
    pta
        Enterprise PTA object.

    posterior_samples
        Posterior samples accepted by reconstruct_smooth_pol_gp().

    psr_name : str
        Example:
            "J1022+1001"

    band : str
        Example:
            "40CM"

    N : int
        Number of posterior GP reconstructions.

    n_grid : int
        Number of points on the fine time grid.

    selection : {"random", "highest", "first"}

    seed : int
        The same seed is deliberately used for X/Y/Z so the
        same posterior hyperparameter samples are reconstructed.

    coefficient_mode : {"mean", "sample"}

    credible_level : float
        Pointwise credible interval for each process.

    Returns
    -------
    result : dict

        result["X"]
        result["Y"]
        result["Z"]

            Complete results returned by
            reconstruct_smooth_pol_gp().

        result["total"]["gp_us"]

            Sample-by-sample sum, shape (N, n_grid).

        result["total"]["median_us"]
        result["total"]["lower_us"]
        result["total"]["upper_us"]

            Posterior summaries of the total process.

        result["mjd"]
            Common dense time grid.
    """

    if not 0.0 < credible_level < 1.0:
        raise ValueError(
            "credible_level must be between 0 and 1."
        )

    results = {}

    # ========================================================
    # Reconstruct X, Y, Z
    # ========================================================

    for axis in ("X", "Y", "Z"):

        process_name = (
            f"{psr_name}_pol_cal_"
            f"{axis.lower()}_{band}"
        )

        print(
            f"Reconstructing {process_name} ..."
        )

        results[axis] = (
            reconstruct_smooth_pol_gp(
                pta=pta,
                posterior_samples=posterior_samples,
                process_name=process_name,
                N=N,
                n_grid=n_grid,
                selection=selection,
                seed=seed,
                flagname=flagname,
                coefficient_mode=coefficient_mode,
                phiinv_method=phiinv_method,
                max_gap_days=max_gap_days,
                validate=validate,
            )
        )

    # ========================================================
    # Verify that all three reconstructions correspond to
    # exactly the same posterior samples.
    # ========================================================

    reference_indices = np.asarray(
        results["X"]["selected_indices"]
    )

    reference_mjd = np.asarray(
        results["X"]["mjd"]
    )

    for axis in ("Y", "Z"):

        current_indices = np.asarray(
            results[axis]["selected_indices"]
        )

        if not np.array_equal(
            reference_indices,
            current_indices,
        ):
            raise RuntimeError(
                f"{axis} reconstruction used different "
                "posterior samples from X. "
                "The total process would therefore not be "
                "a valid sample-by-sample sum."
            )

        current_mjd = np.asarray(
            results[axis]["mjd"]
        )

        if (
            current_mjd.shape != reference_mjd.shape
            or not np.allclose(
                current_mjd,
                reference_mjd,
                rtol=0.0,
                atol=1e-10,
                equal_nan=True,
            )
        ):
            raise RuntimeError(
                f"{axis} uses a different time grid from X."
            )

    # ========================================================
    # Individual projected GP samples
    # ========================================================

    gp_x = np.asarray(
        results["X"]["gp_us"]
    )

    gp_y = np.asarray(
        results["Y"]["gp_us"]
    )

    gp_z = np.asarray(
        results["Z"]["gp_us"]
    )

    if not (
        gp_x.shape
        == gp_y.shape
        == gp_z.shape
    ):
        raise RuntimeError(
            "X/Y/Z GP arrays have different shapes."
        )

    # ========================================================
    # TOTAL PROCESS
    #
    # Important:
    #
    # Sum realizations FIRST:
    #
    #     g_total^(s) =
    #         g_X^(s) + g_Y^(s) + g_Z^(s)
    #
    # then calculate posterior quantiles.
    # ========================================================

    gp_total = (
        gp_x
        + gp_y
        + gp_z
    )

    alpha = (
        1.0 - credible_level
    ) / 2.0

    quantiles = [
        alpha,
        0.5,
        1.0 - alpha,
    ]

    # Individual summaries
    for axis in ("X", "Y", "Z"):

        gp = np.asarray(
            results[axis]["gp_us"]
        )

        lower, median, upper = (
            np.nanquantile(
                gp,
                quantiles,
                axis=0,
            )
        )

        results[axis]["lower_us"] = lower
        results[axis]["median_us"] = median
        results[axis]["upper_us"] = upper

    # Total summary
    (
        total_lower,
        total_median,
        total_upper,
    ) = np.nanquantile(
        gp_total,
        quantiles,
        axis=0,
    )

    # ========================================================
    # Also sum the exact GP at observed TOAs
    # ========================================================

    observed_total_us = (
        np.asarray(
            results["X"]["observed_gp_us"]
        )
        +
        np.asarray(
            results["Y"]["observed_gp_us"]
        )
        +
        np.asarray(
            results["Z"]["observed_gp_us"]
        )
    )

    observed_total_median_us = (
        np.nanmedian(
            observed_total_us,
            axis=0,
        )
    )

    # ========================================================
    # Package total result
    # ========================================================

    results["total"] = {
        "process_name": (
            f"{psr_name}_pol_cal_total_{band}"
        ),

        "gp_us": gp_total,

        "median_us": total_median,
        "lower_us": total_lower,
        "upper_us": total_upper,

        "observed_gp_us":
            observed_total_us,

        "observed_median_us":
            observed_total_median_us,

        "selected_indices":
            reference_indices,

        "credible_level":
            credible_level,
    }

    results["mjd"] = reference_mjd
    results["observed_mjd"] = np.asarray(
        results["X"]["observed_mjd"]
    )

    results["band_mask"] = np.asarray(
        results["X"]["band_mask"]
    )

    results["psr_name"] = psr_name
    results["band"] = band

    return results



def plot_pol_xyz_total(
    pta,
    posterior_samples,
    psr_name,
    band,
    psr_object,
    *,
    N=100,
    n_grid=5000,
    selection="random",
    seed=1234,
    flagname="B",
    coefficient_mode="mean",
    phiinv_method="cliques",
    max_gap_days=None,
    validate=True,
    credible_level=0.90,
    max_curves=30,
    curve_alpha=0.055,
    show_observed_gp=True,
    show_residuals=True,

    # ========================================================
    # Layout / projected timing-delay reconstruction
    # ========================================================
    plot_xyz_together=False,
    show_projected_reconstruction=True,
    show_total=True,
    component_colors=None,

    # ========================================================
    # Underlying boost / distortion process b(t)
    # ========================================================
    show_boost_process=False,
    show_total_boost=False,
    show_boost_interval=True,
    boost_linewidth=1.5,
    boost_alpha=0.90,
    boost_ylim=None,

    # Projected timing-delay limits
    ylim=None,

    # Figure sizing
    figsize=None,

    # Residual appearance
    residual_marker="+",
    residual_markersize=5,
    residual_marker_alpha=0.70,
    residual_error_alpha=0.10,
    residual_elinewidth=0.6,
    residual_markeredgewidth=0.8,
):
    """
    Plot the X, Y, Z polarization reconstructions and optional totals.

    Two physically different quantities can be displayed:

        b_axis(t)
            Underlying polarization boost/distortion process.
            DIMENSIONLESS.

        Delta_axis(t) * b_axis(t)
            Arrival-time perturbation produced by that component.
            Units of seconds, plotted here in microseconds.

    Parameters
    ----------
    plot_xyz_together : bool, default=False
        If False, X/Y/Z are shown in separate panels. If True, they are
        overplotted in one panel.

    show_projected_reconstruction : bool, default=True
        Plot the distortion-projected timing-delay reconstruction
        Delta_axis(t) * b_axis(t).

        If False, the projected reconstruction and timing residuals are not
        plotted. The primary axes instead show b_X(t), b_Y(t), b_Z(t)
        directly, so this is the switch to use for a boost-only figure.

    show_total : bool, default=True
        Plot the projected total timing delay X + Y + Z. This option only
        applies when show_projected_reconstruction=True.

    show_boost_process : bool, default=False
        When projected timing delays are shown, overplot b_X(t), b_Y(t),
        b_Z(t) on secondary y-axes. In boost-only mode
        (show_projected_reconstruction=False), X/Y/Z boosts are always shown
        on the primary axes, so this flag is not required.

    show_total_boost : bool, default=False
        Plot the sample-by-sample boost-vector magnitude

            |b(t)| = sqrt(b_X(t)^2 + b_Y(t)^2 + b_Z(t)^2)

        before calculating its posterior credible interval. The argument name
        is retained for backwards compatibility. The boost magnitude is
        dimensionless, non-negative, and is not multiplied by 1e6.

    show_boost_interval : bool, default=True
        Show the pointwise credible interval for boost processes.

    curve_alpha : float, default=0.055
        Opacity of the individual posterior realization curves for both
        projected timing delays and boost processes. Must be between 0 and 1.

    component_colors : dict or None
        Colors used for X/Y/Z/total.

    psr_object
        Filtered pulsar object with .toas, .residuals, and .toaerrs. These
        are only used when show_projected_reconstruction=True and
        show_residuals=True.

    Returns
    -------
    fig
    axes
        ndarray of primary matplotlib axes.
    reconstruction
        Output from reconstruct_pol_xyz_total(), augmented with the boost
        magnitude under reconstruction["total"]["boost_process"] and
        reconstruction["boost_magnitude"].
    boost_axes
        Dictionary containing the axes on which boost processes were drawn.
        In projected mode these are usually secondary axes; in boost-only
        mode they refer to the primary axes.
    """

    if component_colors is None:
        component_colors = {
            "X": "tab:blue",
            "Y": "tab:orange",
            "Z": "tab:green",
            "total": "black",
        }
    else:
        component_colors = dict(component_colors)
        defaults = {
            "X": "tab:blue",
            "Y": "tab:orange",
            "Z": "tab:green",
            "total": "black",
        }
        for key, value in defaults.items():
            component_colors.setdefault(key, value)

    curve_alpha = float(curve_alpha)
    if not np.isfinite(curve_alpha) or not 0.0 <= curve_alpha <= 1.0:
        raise ValueError("curve_alpha must be between 0 and 1.")

    # ========================================================
    # Reconstruction
    # ========================================================

    reconstruction = reconstruct_pol_xyz_total(
        pta=pta,
        posterior_samples=posterior_samples,
        psr_name=psr_name,
        band=band,
        N=N,
        n_grid=n_grid,
        selection=selection,
        seed=seed,
        flagname=flagname,
        coefficient_mode=coefficient_mode,
        phiinv_method=phiinv_method,
        max_gap_days=max_gap_days,
        validate=validate,
        credible_level=credible_level,
    )

    mjd = np.asarray(reconstruction["mjd"], dtype=float)
    interval_percent = 100.0 * credible_level
    alpha_q = 0.5 * (1.0 - credible_level)
    quantiles = [alpha_q, 0.5, 1.0 - alpha_q]

    # ========================================================
    # Underlying boost-vector magnitude, evaluated realization-by-realization.
    #
    # Do not take the magnitude of component-wise posterior medians: that
    # would discard the joint posterior structure and give incorrect
    # credible intervals.
    # ========================================================

    boost_x = np.asarray(reconstruction["X"]["boost_process"], dtype=float)
    boost_y = np.asarray(reconstruction["Y"]["boost_process"], dtype=float)
    boost_z = np.asarray(reconstruction["Z"]["boost_process"], dtype=float)

    if not (boost_x.shape == boost_y.shape == boost_z.shape):
        raise RuntimeError("X/Y/Z boost arrays have different shapes.")

    boost_total = np.sqrt(
        boost_x**2
        + boost_y**2
        + boost_z**2
    )


    boost_total_lower, boost_total_median, boost_total_upper = np.nanquantile(
        boost_total,
        quantiles,
        axis=0,
    )

    reconstruction["total"]["boost_process"] = boost_total
    reconstruction["total"]["b_process"] = boost_total
    reconstruction["total"]["boost_lower"] = boost_total_lower
    reconstruction["total"]["boost_median"] = boost_total_median
    reconstruction["total"]["boost_upper"] = boost_total_upper

    # Explicit, physically descriptive alias. The entries under "total" are
    # kept so existing plotting calls and downstream notebooks continue to
    # work when show_total_boost=True.
    reconstruction["boost_magnitude"] = {
        "process_name": f"{psr_name}_pol_cal_boost_magnitude_{band}",
        "boost_process": boost_total,
        "b_process": boost_total,
        "boost_lower": boost_total_lower,
        "boost_median": boost_total_median,
        "boost_upper": boost_total_upper,
        "selected_indices": np.asarray(
            reconstruction["total"]["selected_indices"]
        ),
        "credible_level": credible_level,
    }

    # Residuals have timing units, so never put them on a boost-only axis.
    plot_residuals = bool(show_residuals and show_projected_reconstruction)

    # ========================================================
    # Residual data
    # ========================================================

    if plot_residuals:
        for attribute in ("toas", "residuals", "toaerrs"):
            if not hasattr(psr_object, attribute):
                raise AttributeError(
                    f"psr_object has no {attribute!r} attribute."
                )

        residual_mjd = np.asarray(psr_object.toas, dtype=float) / 86400.0
        residual_us = np.asarray(psr_object.residuals, dtype=float) * 1e6
        residual_error_us = np.asarray(psr_object.toaerrs, dtype=float) * 1e6

        if not (
            residual_mjd.size
            == residual_us.size
            == residual_error_us.size
        ):
            raise ValueError(
                "TOA, residual and TOA-error arrays must have the same length."
            )

        order = np.argsort(residual_mjd)
        residual_mjd = residual_mjd[order]
        residual_us = residual_us[order]
        residual_error_us = residual_error_us[order]

    # ========================================================
    # Small plotting helpers
    # ========================================================

    def _set_limits(ax, limits):
        if limits is None:
            return
        if np.isscalar(limits):
            limit = abs(float(limits))
            ax.set_ylim(-limit, limit)
        else:
            if len(limits) != 2:
                raise ValueError(
                    "Axis limits must be a scalar, two-element sequence, or None."
                )
            ax.set_ylim(float(limits[0]), float(limits[1]))

    def _plot_residuals(ax, add_label=True):
        if not plot_residuals:
            return

        marker_rgba = to_rgba("black", residual_marker_alpha)
        error_rgba = to_rgba("black", residual_error_alpha)

        ax.errorbar(
            residual_mjd,
            residual_us,
            yerr=residual_error_us,
            fmt=residual_marker,
            color=marker_rgba,
            ecolor=error_rgba,
            markersize=residual_markersize,
            markeredgewidth=residual_markeredgewidth,
            elinewidth=residual_elinewidth,
            capsize=0,
            linestyle="none",
            label=(f"{band} residuals" if add_label else None),
            zorder=0,
        )

    def _curve_indices(draws):
        if max_curves is None or max_curves <= 0:
            return np.array([], dtype=int)
        n_show = min(int(max_curves), draws.shape[0])
        return np.unique(
            np.linspace(0, draws.shape[0] - 1, n_show, dtype=int)
        )

    def _with_interval_legend(handles, labels, visible):
        """Prepend one neutral legend patch for all credible intervals."""

        if not visible:
            return handles, labels

        interval_label = f"{interval_percent:.0f}% interval"
        interval_patch = Patch(
            facecolor="0.70",
            edgecolor="0.45",
            alpha=0.45,
        )
        return (
            [interval_patch, *handles],
            [interval_label, *labels],
        )

    def _observed_component_median(key):
        if key == "total":
            return np.asarray(
                reconstruction["total"]["observed_median_us"], dtype=float
            )
        return np.nanmedian(
            np.asarray(reconstruction[key]["observed_gp_us"], dtype=float),
            axis=0,
        )

    def _plot_projected_component(
        ax,
        key,
        *,
        show_interval=True,
        show_curves=True,
        show_points=True,
        linewidth=2.0,
        label_prefix=None,
    ):
        result = reconstruction[key]
        gp_us = np.asarray(result["gp_us"], dtype=float)
        color = component_colors[key]

        if label_prefix is None:
            label_prefix = key

        if show_curves:
            for index in _curve_indices(gp_us):
                ax.plot(
                    mjd,
                    gp_us[index],
                    color=color,
                    linewidth=0.65,
                    alpha=curve_alpha,
                    zorder=1,
                )

        if show_interval:
            ax.fill_between(
                mjd,
                result["lower_us"],
                result["upper_us"],
                color=color,
                alpha=(0.14 if key != "total" else 0.10),
                label="_nolegend_",
                zorder=2,
            )

        ax.plot(
            mjd,
            result["median_us"],
            color=color,
            linewidth=linewidth,
            label=(
                f"{label_prefix} projected median"
                if plot_xyz_together
                else "Projected GP median"
            ),
            zorder=4,
        )

        if show_points and show_observed_gp:
            observed_mjd = np.asarray(reconstruction["observed_mjd"], dtype=float)
            band_mask = np.asarray(reconstruction["band_mask"], dtype=bool)
            observed_median = _observed_component_median(key)

            ax.scatter(
                observed_mjd[band_mask],
                observed_median[band_mask],
                s=(7 if key != "total" else 9),
                color=color,
                alpha=(0.55 if key != "total" else 0.65),
                label=(
                    f"{label_prefix} GP at TOAs"
                    if plot_xyz_together
                    else "Projected GP at TOAs"
                ),
                zorder=5,
            )

    def _boost_draws(key):
        return np.asarray(reconstruction[key]["boost_process"], dtype=float)

    def _boost_label(key):
        if key == "total":
            return r"$|\mathbf{b}(t)|$"
        return rf"$b_{key}(t)$"

    def _plot_boost_component(
        boost_ax,
        key,
        *,
        add_interval=True,
        show_curves=False,
        linewidth=None,
    ):
        """Plot the dimensionless underlying boost process b(t)."""

        boost_draws = _boost_draws(key)
        boost_lower, boost_median, boost_upper = np.nanquantile(
            boost_draws,
            quantiles,
            axis=0,
        )
        color = component_colors[key]

        if show_curves:
            for index in _curve_indices(boost_draws):
                boost_ax.plot(
                    mjd,
                    boost_draws[index],
                    color=color,
                    linewidth=0.65,
                    alpha=curve_alpha,
                    zorder=1,
                )

        if add_interval and show_boost_interval:
            boost_ax.fill_between(
                mjd,
                boost_lower,
                boost_upper,
                color=color,
                alpha=(0.12 if key != "total" else 0.09),
                label="_nolegend_",
                zorder=2,
            )

        boost_ax.plot(
            mjd,
            boost_median,
            color=color,
            linestyle="--",
            linewidth=(
                linewidth
                if linewidth is not None
                else (boost_linewidth if key != "total" else 1.35 * boost_linewidth)
            ),
            alpha=boost_alpha,
            label=_boost_label(key),
            zorder=3,
        )

    boost_axes = {
        "X": None,
        "Y": None,
        "Z": None,
        "total": None,
        "overlay": None,
    }

    # ========================================================
    # OVERLAY MODE: X/Y/Z on a single panel
    # ========================================================

    if plot_xyz_together:
        if figsize is None:
            figsize = (14, 5.5)

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        axes = np.asarray([ax])

        if show_projected_reconstruction:
            _plot_residuals(ax)

            for key in ("X", "Y", "Z"):
                _plot_projected_component(
                    ax,
                    key,
                    show_interval=True,
                    show_curves=True,
                    show_points=show_observed_gp,
                    linewidth=2.0,
                    label_prefix=key,
                )

            if show_total:
                _plot_projected_component(
                    ax,
                    "total",
                    show_interval=True,
                    show_curves=False,
                    show_points=show_observed_gp,
                    linewidth=2.7,
                    label_prefix="Total",
                )

            ax.set_ylabel(r"Polarization delay ($\mu$s)")
            _set_limits(ax, ylim)

            if show_boost_process or show_total_boost:
                boost_ax = ax.twinx()
                boost_axes["overlay"] = boost_ax

                if show_boost_process:
                    for key in ("X", "Y", "Z"):
                        boost_axes[key] = boost_ax
                        _plot_boost_component(boost_ax, key)

                if show_total_boost:
                    boost_axes["total"] = boost_ax
                    _plot_boost_component(
                        boost_ax,
                        "total",
                        linewidth=1.35 * boost_linewidth,
                    )

                boost_ax.set_ylabel(
                    r"$\vec{b}(t)$"
                )
                _set_limits(boost_ax, boost_ylim)

        else:
            # Boost-only mode: put b_X, b_Y, b_Z directly on the primary axis.
            boost_axes["overlay"] = ax
            for key in ("X", "Y", "Z"):
                boost_axes[key] = ax
                _plot_boost_component(
                    ax,
                    key,
                    show_curves=True,
                    linewidth=boost_linewidth,
                )

            if show_total_boost:
                boost_axes["total"] = ax
                _plot_boost_component(
                    ax,
                    "total",
                    show_curves=False,
                    linewidth=1.35 * boost_linewidth,
                )

            ax.set_ylabel(r"$\vec{b}(t)$")
            _set_limits(ax, boost_ylim)

        ax.axhline(0.0, color="grey", linewidth=0.8, alpha=0.4, zorder=-1)
        ax.set_xlabel("MJD")
        # ax.set_title(
        #     (
        #         f"{psr_name}: X, Y, Z polarization reconstruction — {band}"
        #         if show_projected_reconstruction
        #         else f"{psr_name}: X, Y, Z boost reconstruction — {band}"
        #     )
        # )

        handles, labels = ax.get_legend_handles_labels()
        if show_projected_reconstruction and boost_axes["overlay"] is not None:
            handles2, labels2 = boost_axes["overlay"].get_legend_handles_labels()
            handles += handles2
            labels += labels2

        interval_visible = (
            show_projected_reconstruction
            or (
                show_boost_interval
                and (
                    not show_projected_reconstruction
                    or show_boost_process
                    or show_total_boost
                )
            )
        )
        handles, labels = _with_interval_legend(
            handles,
            labels,
            interval_visible,
        )

        if handles:
            ax.legend(handles, labels, loc="best", ncol=4)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        return fig, axes, reconstruction, boost_axes

    # ========================================================
    # SEPARATE-PANEL MODE
    # ========================================================

    panel_names = ["X", "Y", "Z"]

    if show_projected_reconstruction:
        if show_total or show_total_boost:
            panel_names.append("total")
    elif show_total_boost:
        panel_names.append("total")

    if figsize is None:
        figsize = (14, 3.1 * len(panel_names))

    fig, axes = plt.subplots(
        len(panel_names),
        1,
        figsize=figsize,
        sharex=True,
        squeeze=False,
    )
    axes = axes[:, 0]

    panel_titles = {
        "X": f"{psr_name}: X — {band}",
        "Y": f"{psr_name}: Y — {band}",
        "Z": f"{psr_name}: Z — {band}",
        "total": (
            f"{psr_name}: X + Y + Z — {band}"
            if show_projected_reconstruction and show_total
            else f"{psr_name}: boost magnitude — {band}"
        ),
    }

    for ax, key in zip(axes, panel_names):
        panel_has_projected = bool(
            show_projected_reconstruction
            and (key != "total" or show_total)
        )

        if panel_has_projected:
            _plot_residuals(ax)
            _plot_projected_component(
                ax,
                key,
                show_interval=True,
                show_curves=True,
                show_points=show_observed_gp,
                linewidth=(2.0 if key != "total" else 2.5),
                label_prefix=key,
            )
            ax.set_ylabel(r"Polarization delay ($\mu$s)")
            _set_limits(ax, ylim)

            # Add boosts on a secondary axis when timing delays occupy primary.
            plot_this_boost = (
                (key in ("X", "Y", "Z") and show_boost_process)
                or (key == "total" and show_total_boost)
            )

            if plot_this_boost:
                boost_ax = ax.twinx()
                boost_axes[key] = boost_ax
                _plot_boost_component(
                    boost_ax,
                    key,
                    show_curves=False,
                    linewidth=(
                        boost_linewidth
                        if key != "total"
                        else 1.35 * boost_linewidth
                    ),
                )
                boost_ax.set_ylabel(
                    rf"$b_{key}(t)$"
                    if key != "total"
                    else r"$|\mathbf{b}(t)|$"
                )
                _set_limits(boost_ax, boost_ylim)

        else:
            # Boost-only panel, used globally when projected plotting is off,
            # and for a total-boost-only panel if show_total=False.
            boost_axes[key] = ax
            _plot_boost_component(
                ax,
                key,
                show_curves=(key != "total"),
                linewidth=(
                    boost_linewidth
                    if key != "total"
                    else 1.35 * boost_linewidth
                ),
            )
            ax.set_ylabel(
                rf"$b_{key}(t)$"
                if key != "total"
                else r"$|\mathbf{b}(t)|$"
            )
            _set_limits(ax, boost_ylim)

        ax.axhline(0.0, color="grey", linewidth=0.8, alpha=0.4, zorder=-1)
        # ax.set_title(panel_titles[key])
        ax.grid(alpha=0.25)

        handles, labels = ax.get_legend_handles_labels()
        if panel_has_projected and boost_axes[key] is not None:
            handles2, labels2 = boost_axes[key].get_legend_handles_labels()
            handles += handles2
            labels += labels2

        handles, labels = _with_interval_legend(
            handles,
            labels,
            panel_has_projected or show_boost_interval,
        )

        if handles:
            ax.legend(handles, labels, loc="best", ncol=3)

    axes[-1].set_xlabel("MJD")

    fig.suptitle(
        (
            f"{psr_name}: polarization GP reconstruction — {band}"
            if show_projected_reconstruction
            else f"{psr_name}: polarization boost reconstruction — {band}"
        ),
        y=0.995,
    )

    fig.tight_layout()
    return fig, axes, reconstruction, boost_axes


def _signal_names_for_pulsar(pta, psr_name):
    """Return all Enterprise signal names belonging to one pulsar."""

    prefix = str(psr_name) + "_"
    return sorted({
        signal.name
        for model in pta.pulsarmodels
        for signal in model._signals
        if signal.name.startswith(prefix)
    })


def _resolve_standard_process_name(
    pta,
    psr_name,
    process_kind,
    process_name_overrides=None,
):
    """Resolve RN, DM or GWB signal names, returning None when absent."""

    process_kind = str(process_kind).strip().lower()
    aliases = {
        "red_noise": "rn",
        "red": "rn",
        "rn": "rn",
        "dispersion_measure": "dm",
        "dm_gp": "dm",
        "dm": "dm",
        "gwb_hd": "gw",
        "gwb": "gw",
        "gw": "gw",
    }

    if process_kind not in aliases:
        raise ValueError(
            f"Unknown process kind {process_kind!r}. "
            "Use 'rn', 'dm', or 'gw'."
        )

    process_kind = aliases[process_kind]
    overrides = (
        {}
        if process_name_overrides is None
        else dict(process_name_overrides)
    )

    if process_kind in overrides:
        requested = str(overrides[process_kind])
        return (
            requested
            if requested in _signal_names_for_pulsar(pta, psr_name)
            else None
        )

    exact_names = {
        "rn": f"{psr_name}_red_noise",
        "dm": f"{psr_name}_dm_gp",
        "gw": f"{psr_name}_gwb_hd",
    }
    signal_names = _signal_names_for_pulsar(pta, psr_name)
    exact_name = exact_names[process_kind]

    if exact_name in signal_names:
        return exact_name

    if process_kind == "rn":
        candidates = [
            name for name in signal_names
            if "red_noise" in name.lower()
        ]
    elif process_kind == "dm":
        candidates = [
            name for name in signal_names
            if (
                name.lower().endswith("_dm_gp")
                or "dm_gp" in name.lower()
            )
        ]
    else:
        candidates = [
            name for name in signal_names
            if "gwb" in name.lower()
        ]

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) > 1:
        raise RuntimeError(
            f"Multiple {process_kind.upper()} signals were found for "
            f"{psr_name}: {candidates}. Supply process_name_overrides."
        )

    return None


def _extract_signal_labels(signal, psr_name, n_columns):
    """Extract numeric Fourier labels from local or common GP signals."""

    raw_labels = signal._labels

    if not isinstance(raw_labels, dict):
        labels = np.asarray(raw_labels, dtype=float).ravel()
        return labels

    # Enterprise common-basis signals commonly retain labels in a dictionary
    # keyed by pulsar name. Prefer exact keys before trying structural
    # fallbacks for version-to-version compatibility.
    preferred_keys = (
        psr_name,
        str(psr_name),
        signal.name,
        getattr(signal, "_psrname", None),
    )

    for key in preferred_keys:
        if key is not None and key in raw_labels:
            try:
                labels = np.asarray(raw_labels[key], dtype=float).ravel()
            except (TypeError, ValueError):
                continue

            if labels.size == n_columns:
                return labels

    numeric_candidates = []

    def collect_numeric_leaves(value, path=()):
        if isinstance(value, dict):
            for key, child in value.items():
                collect_numeric_leaves(child, path + (str(key),))
            return

        try:
            candidate = np.asarray(value, dtype=float).ravel()
        except (TypeError, ValueError):
            return

        if candidate.size == n_columns:
            numeric_candidates.append((path, candidate))

    collect_numeric_leaves(raw_labels)

    psr_matches = [
        candidate
        for path, candidate in numeric_candidates
        if any(str(psr_name) == part for part in path)
    ]

    if len(psr_matches) == 1:
        return psr_matches[0]

    if len(numeric_candidates) == 1:
        return numeric_candidates[0][1]

    available_paths = [
        "/".join(path) if path else "<root>"
        for path, _ in numeric_candidates
    ]
    raise RuntimeError(
        f"Could not select {n_columns} Fourier labels for {signal.name!r} "
        f"and pulsar {psr_name!r} from dictionary-valued _labels. "
        f"Candidate paths: {available_paths}; top-level keys: "
        f"{list(raw_labels.keys())}."
    )


def reconstruct_smooth_standard_gp(
    pta,
    posterior_samples,
    process_name,
    psr_object,
    *,
    process_kind,
    N=100,
    n_grid=5000,
    grid_mjd=None,
    selection="random",
    seed=1234,
    coefficient_mode="mean",
    phiinv_method="cliques",
    credible_level=0.90,
    dm_frequency_mhz=1400.0,
    validate=True,
):
    """
    Reconstruct a standard Fourier-basis RN, DM, or GWB process.

    RN and GWB are reconstructed as achromatic timing delays. The DM process
    is reconstructed at ``dm_frequency_mhz``; 1400 MHz corresponds to the
    conventional reference-frequency realization when Enterprise uses the
    standard ``(1400 MHz / nu)^2`` DM basis scaling.
    """

    process_kind = str(process_kind).strip().lower()
    if process_kind not in {"rn", "dm", "gw"}:
        raise ValueError("process_kind must be 'rn', 'dm', or 'gw'.")

    if not 0.0 < credible_level < 1.0:
        raise ValueError("credible_level must be between 0 and 1.")

    signal = _find_signal(pta, process_name)
    original_toas = np.asarray(psr_object.toas, dtype=float)
    original_mjd = original_toas / 86400.0

    samples, selected_indices = _select_posterior_samples(
        posterior_samples,
        N=N,
        selection=selection,
        seed=seed,
    )

    first_params = pta.map_params(samples[selected_indices[0]])
    F_enterprise = np.asarray(
        signal.get_basis(params=first_params),
        dtype=float,
    )
    labels = _extract_signal_labels(
        signal=signal,
        psr_name=getattr(psr_object, "name", process_name.split("_")[0]),
        n_columns=F_enterprise.shape[1],
    )

    if F_enterprise.shape[0] != original_toas.size:
        raise RuntimeError(
            f"{process_name} basis has {F_enterprise.shape[0]} rows, but "
            f"psr_object contains {original_toas.size} TOAs. Pass the same "
            "unfiltered pulsar object used to build the PTA."
        )

    if labels.size != F_enterprise.shape[1] or labels.size % 2:
        raise RuntimeError(
            f"{process_name} does not have valid sine/cosine Fourier labels."
        )

    if not np.allclose(labels[::2], labels[1::2]):
        raise RuntimeError(
            f"{process_name} labels are not arranged as Fourier pairs."
        )

    frequencies = labels[::2].copy()
    n_modes = frequencies.size

    if process_kind == "dm":
        if not hasattr(psr_object, "freqs"):
            raise AttributeError(
                "psr_object must contain observing frequencies in .freqs "
                "to reconstruct the DM GP."
            )

        observed_freqs_mhz = np.asarray(psr_object.freqs, dtype=float)
        if observed_freqs_mhz.size != original_toas.size:
            raise ValueError(
                "psr_object.freqs and psr_object.toas have different lengths."
            )

        if not np.isfinite(dm_frequency_mhz) or dm_frequency_mhz <= 0.0:
            raise ValueError("dm_frequency_mhz must be positive and finite.")

        observed_weight = (1400.0 / observed_freqs_mhz) ** 2
        grid_weight = (1400.0 / float(dm_frequency_mhz)) ** 2
    else:
        observed_weight = np.ones(original_toas.size, dtype=float)
        grid_weight = 1.0

    phases = np.zeros(n_modes, dtype=float)
    weight_tolerance = 1e-14

    for mode_index, freq in enumerate(frequencies):
        fsin = F_enterprise[:, 2 * mode_index]
        fcos = F_enterprise[:, 2 * mode_index + 1]
        usable = (
            np.isfinite(observed_weight)
            & (np.abs(observed_weight) > weight_tolerance)
        )
        indices = np.flatnonzero(usable)

        if indices.size == 0:
            raise RuntimeError(
                f"No usable basis weights for {process_name}."
            )

        measured_angle = np.arctan2(
            fsin[indices] / observed_weight[indices],
            fcos[indices] / observed_weight[indices],
        )
        phase_samples = (
            measured_angle
            - 2.0 * np.pi * original_toas[indices] * freq
        )
        phases[mode_index] = np.angle(
            np.mean(np.exp(1j * phase_samples))
        )

    observed_angle = (
        2.0 * np.pi
        * original_toas[:, None]
        * frequencies[None, :]
        + phases[None, :]
    )
    F_latent_observed = np.empty(
        (original_toas.size, 2 * n_modes),
        dtype=float,
    )
    F_latent_observed[:, ::2] = np.sin(observed_angle)
    F_latent_observed[:, 1::2] = np.cos(observed_angle)
    F_reconstructed = observed_weight[:, None] * F_latent_observed
    basis_max_error = float(
        np.nanmax(np.abs(F_reconstructed - F_enterprise))
    )

    if validate and not np.allclose(
        F_reconstructed,
        F_enterprise,
        rtol=1e-7,
        atol=1e-12,
    ):
        raise RuntimeError(
            f"Could not reproduce the fitted basis for {process_name}. "
            f"Maximum basis error = {basis_max_error:.6g}."
        )

    if grid_mjd is None:
        grid_mjd = np.linspace(
            float(np.nanmin(original_mjd)),
            float(np.nanmax(original_mjd)),
            int(n_grid),
        )
    else:
        grid_mjd = np.asarray(grid_mjd, dtype=float)

    if grid_mjd.ndim != 1 or grid_mjd.size < 2:
        raise ValueError("grid_mjd must be a one-dimensional time grid.")

    grid_toas = grid_mjd * 86400.0
    grid_angle = (
        2.0 * np.pi
        * grid_toas[:, None]
        * frequencies[None, :]
        + phases[None, :]
    )
    F_grid = np.empty(
        (grid_mjd.size, 2 * n_modes),
        dtype=float,
    )
    F_grid[:, ::2] = np.sin(grid_angle)
    F_grid[:, 1::2] = np.cos(grid_angle)
    F_grid *= grid_weight

    conditional_gp = utils.ConditionalGP(
        pta,
        phiinv_method=phiinv_method,
    )
    reconstructions = []
    observed_reconstructions = []
    coefficient_samples = []

    for posterior_index in selected_indices:
        params = pta.map_params(samples[posterior_index])

        if coefficient_mode == "mean":
            coefficient_dict = conditional_gp.get_mean_coefficients(params)
        elif coefficient_mode == "sample":
            sampled = conditional_gp.sample_coefficients(params, n=1)
            coefficient_dict = (
                sampled
                if isinstance(sampled, dict)
                else sampled[0]
            )
        else:
            raise ValueError(
                "coefficient_mode must be 'mean' or 'sample'."
            )

        coeff = _get_coefficient_vector(coefficient_dict, process_name)
        if coeff.size != F_grid.shape[1]:
            raise RuntimeError(
                f"ConditionalGP returned {coeff.size} coefficients for "
                f"{process_name}, but its basis has {F_grid.shape[1]} columns."
            )

        reconstructions.append(F_grid @ coeff)
        observed_reconstructions.append(F_enterprise @ coeff)
        coefficient_samples.append(coeff)

    reconstructions = np.asarray(reconstructions, dtype=float)
    observed_reconstructions = np.asarray(
        observed_reconstructions,
        dtype=float,
    )
    coefficient_samples = np.asarray(coefficient_samples, dtype=float)

    alpha_q = 0.5 * (1.0 - credible_level)
    lower_us, median_us, upper_us = np.nanquantile(
        reconstructions * 1e6,
        [alpha_q, 0.5, 1.0 - alpha_q],
        axis=0,
    )

    return {
        "mjd": grid_mjd,
        "gp_seconds": reconstructions,
        "gp_us": reconstructions * 1e6,
        "lower_us": lower_us,
        "median_us": median_us,
        "upper_us": upper_us,
        "observed_mjd": original_mjd,
        "observed_gp_seconds": observed_reconstructions,
        "observed_gp_us": observed_reconstructions * 1e6,
        "frequencies": frequencies,
        "phases": phases,
        "coefficients": coefficient_samples,
        "selected_indices": selected_indices,
        "process_name": process_name,
        "process_kind": process_kind,
        "dm_frequency_mhz": (
            float(dm_frequency_mhz)
            if process_kind == "dm"
            else None
        ),
        "basis_max_error": basis_max_error,
        "credible_level": credible_level,
    }



def _find_linear_timing_model_signal(
    pta,
    psr_name,
    signal_name=None,
):
    """
    Find the marginalized Enterprise linear timing-model basis signal for one
    pulsar.

    Parameters
    ----------
    pta
        Enterprise PTA object.

    psr_name : str
        Pulsar name, for example ``"J1022+1001"``.

    signal_name : str or None
        Optional explicit signal name. This may be either the full Enterprise
        signal name, e.g. ``"J1022+1001_linear_timing_model"``, or the signal
        suffix ``"linear_timing_model"``. When omitted, the function searches
        for the basis signal whose Enterprise ``signal_name`` is
        ``"linear timing model"`` or whose name contains ``"timing_model"``.

    Returns
    -------
    signal
        Enterprise timing-model basis signal.
    """

    psr_name = str(psr_name)

    candidates = []

    for model in pta.pulsarmodels:
        for signal in model._signals:
            if signal.signal_type not in ("basis", "common basis"):
                continue

            full_name = str(signal.name)

            if not full_name.startswith(psr_name + "_"):
                continue

            if signal_name is not None:
                requested = str(signal_name)
                if full_name == requested or full_name == f"{psr_name}_{requested}":
                    candidates.append(signal)
                continue

            descriptive_name = str(
                getattr(signal, "signal_name", "")
            ).strip().lower()

            if (
                descriptive_name == "linear timing model"
                or "timing_model" in full_name.lower()
                or "timing model" in descriptive_name
            ):
                candidates.append(signal)

    if len(candidates) == 1:
        return candidates[0]

    available = sorted(
        str(signal.name)
        for model in pta.pulsarmodels
        for signal in model._signals
        if (
            signal.signal_type in ("basis", "common basis")
            and str(signal.name).startswith(psr_name + "_")
        )
    )

    if not candidates:
        raise KeyError(
            f"Could not find a linear timing-model basis signal for "
            f"{psr_name!r}.\n"
            f"Available basis signals for this pulsar:\n"
            + "\n".join(available)
        )

    raise RuntimeError(
        f"Found multiple timing-model signals for {psr_name!r}:\n"
        + "\n".join(str(signal.name) for signal in candidates)
        + "\nPass timing_model_signal_name explicitly."
    )


def reconstruct_linear_timing_model(
    pta,
    posterior_samples,
    psr_object,
    *,
    N=100,
    selection="random",
    seed=1234,
    selected_indices=None,
    coefficient_mode="mean",
    phiinv_method="cliques",
    credible_level=0.90,
    timing_model_signal_name=None,
):
    """
    Reconstruct the marginalized linear timing-model contribution at the
    original TOA epochs and use it to form timing-model-corrected residuals.

    The Enterprise likelihood models the input timing residual vector as a
    sum of basis processes. For the ordinary ``gp_signals.TimingModel()``, the
    linear timing-model contribution is

        delta_t_TM = M_TM epsilon,

    where ``M_TM`` is the exact timing-model basis used by Enterprise. This
    function obtains the conditional timing-model process from
    ``utils.ConditionalGP`` so that the same normalized/SVD timing basis and
    the same joint covariance model used in the likelihood are respected.

    The corrected residuals are then

        r_corrected = r_enterprise - delta_t_TM.

    This is *not* a second subtraction of the full TEMPO2 timing model. The
    full timing solution has already been removed when ``psr_object.residuals``
    was constructed; only the posterior linear timing-model correction is
    removed here.

    Parameters
    ----------
    pta
        Enterprise PTA object.

    posterior_samples
        Posterior sample array or poco-style result dictionary accepted by
        ``_select_posterior_samples``.

    psr_object
        The unfiltered Enterprise Pulsar object used to construct the PTA.
        It must provide ``name``, ``toas`` and ``residuals``.

    N : int
        Number of posterior hyperparameter samples when ``selected_indices``
        is not supplied.

    selection : {"random", "highest", "first"}
        Posterior-sample selection rule used when ``selected_indices`` is not
        supplied.

    seed : int
        Random seed used by posterior selection.

    selected_indices : array-like or None
        Explicit posterior sample indices. Supplying these is recommended when
        the timing correction must correspond exactly to a polarization/noise
        reconstruction that has already selected posterior samples.

    coefficient_mode : {"mean", "sample"}
        ``"mean"`` uses the conditional mean timing-model coefficients for
        each posterior hyperparameter sample. ``"sample"`` uses one random
        conditional realization per hyperparameter sample. For plotting
        corrected residuals, ``"mean"`` is normally the appropriate choice.

    phiinv_method : str
        Method passed to ``utils.ConditionalGP``.

    credible_level : float
        Pointwise posterior interval for the timing-model contribution and the
        corrected residuals.

    timing_model_signal_name : str or None
        Optional explicit Enterprise timing-model signal name or suffix.

    Returns
    -------
    result : dict
        Contains the timing-model process, raw residuals, sample-by-sample
        corrected residuals, posterior summaries and selected indices. Timing
        quantities are returned in both seconds and microseconds where useful.
    """

    if not 0.0 < credible_level < 1.0:
        raise ValueError("credible_level must be between 0 and 1.")

    for attribute in ("name", "toas", "residuals"):
        if not hasattr(psr_object, attribute):
            raise AttributeError(
                f"psr_object has no {attribute!r} attribute."
            )

    samples, automatic_indices = _select_posterior_samples(
        posterior_samples,
        N=N,
        selection=selection,
        seed=seed,
    )

    if selected_indices is None:
        selected_indices = np.asarray(
            automatic_indices,
            dtype=int,
        )
    else:
        selected_indices = np.asarray(
            selected_indices,
            dtype=int,
        ).ravel()

        if selected_indices.size == 0:
            raise ValueError("selected_indices must not be empty.")

        if np.any(selected_indices < 0) or np.any(
            selected_indices >= samples.shape[0]
        ):
            raise IndexError(
                "selected_indices contains an index outside the posterior "
                "sample array."
            )

    timing_signal = _find_linear_timing_model_signal(
        pta=pta,
        psr_name=psr_object.name,
        signal_name=timing_model_signal_name,
    )

    first_params = pta.map_params(
        samples[selected_indices[0]]
    )

    timing_basis = timing_signal.get_basis(
        params=first_params
    )

    if timing_basis is None:
        raise RuntimeError(
            f"{timing_signal.name!r} does not expose a marginalized basis. "
            "This reconstruction requires the ordinary Enterprise "
            "gp_signals.TimingModel(coefficients=False), not an explicitly "
            "sampled timing-coefficient signal."
        )

    timing_basis = np.asarray(
        timing_basis,
        dtype=float,
    )

    residuals_seconds = np.asarray(
        psr_object.residuals,
        dtype=float,
    )
    mjd = np.asarray(
        psr_object.toas,
        dtype=float,
    ) / 86400.0

    if timing_basis.shape[0] != residuals_seconds.size:
        raise RuntimeError(
            f"Timing-model basis for {psr_object.name} has "
            f"{timing_basis.shape[0]} rows, but psr_object.residuals has "
            f"{residuals_seconds.size} entries. Pass the same unfiltered "
            "Pulsar object used to build the PTA."
        )

    coefficient_mode = str(coefficient_mode).strip().lower()

    if coefficient_mode not in {"mean", "sample"}:
        raise ValueError(
            "coefficient_mode must be 'mean' or 'sample'."
        )

    conditional_gp = utils.ConditionalGP(
        pta,
        phiinv_method=phiinv_method,
    )

    timing_draws = []

    for posterior_index in selected_indices:
        params = pta.map_params(
            samples[posterior_index]
        )

        if coefficient_mode == "mean":
            process_dict = conditional_gp.get_mean_processes(
                params
            )
        else:
            sampled_processes = conditional_gp.sample_processes(
                params,
                n=1,
            )
            process_dict = (
                sampled_processes
                if isinstance(sampled_processes, dict)
                else sampled_processes[0]
            )

        if timing_signal.name not in process_dict:
            available = sorted(
                str(key)
                for key in process_dict.keys()
            )
            raise KeyError(
                f"ConditionalGP did not return timing process "
                f"{timing_signal.name!r}.\n"
                f"Available process keys:\n"
                + "\n".join(available)
            )

        timing_process = np.asarray(
            process_dict[timing_signal.name],
            dtype=float,
        )

        if timing_process.size != residuals_seconds.size:
            raise RuntimeError(
                f"Conditional timing-model process has "
                f"{timing_process.size} samples, but the pulsar has "
                f"{residuals_seconds.size} TOAs."
            )

        timing_draws.append(
            timing_process
        )

    timing_draws = np.asarray(
        timing_draws,
        dtype=float,
    )

    corrected_draws = (
        residuals_seconds[None, :]
        - timing_draws
    )

    alpha_q = 0.5 * (1.0 - credible_level)
    quantiles = [alpha_q, 0.5, 1.0 - alpha_q]

    timing_lower, timing_median, timing_upper = np.nanquantile(
        timing_draws,
        quantiles,
        axis=0,
    )

    corrected_lower, corrected_median, corrected_upper = np.nanquantile(
        corrected_draws,
        quantiles,
        axis=0,
    )

    timing_mean = np.nanmean(
        timing_draws,
        axis=0,
    )
    corrected_mean = (
        residuals_seconds
        - timing_mean
    )

    return {
        "mjd": mjd,
        "timing_model_seconds": timing_draws,
        "timing_model_us": timing_draws * 1e6,
        "timing_lower_us": timing_lower * 1e6,
        "timing_median_us": timing_median * 1e6,
        "timing_upper_us": timing_upper * 1e6,
        "timing_mean_us": timing_mean * 1e6,
        "raw_residuals_seconds": residuals_seconds,
        "raw_residuals_us": residuals_seconds * 1e6,
        "corrected_residuals_seconds": corrected_draws,
        "corrected_residuals_us": corrected_draws * 1e6,
        "corrected_lower_us": corrected_lower * 1e6,
        "corrected_median_us": corrected_median * 1e6,
        "corrected_upper_us": corrected_upper * 1e6,
        "corrected_mean_us": corrected_mean * 1e6,
        "selected_indices": selected_indices,
        "coefficient_mode": coefficient_mode,
        "timing_signal_name": str(timing_signal.name),
        "credible_level": credible_level,
    }

def plot_total_pol_delay_by_band(
    pta,
    posterior_samples,
    psr_name,
    bands=("10CM", "20CM", "40CM"),
    *,
    quantity="delay",
    N=100,
    n_grid=5000,
    selection="random",
    seed=1234,
    flagname="B",
    coefficient_mode="mean",
    phiinv_method="cliques",
    max_gap_days=None,
    validate=True,
    credible_level=0.90,
    band_colors=None,
    show_interval=True,
    show_sample_curves=False,
    max_curves=20,
    show_observed_gp=False,
    psr_object=None,
    show_residuals=False,
    remove_timing_model=False,
    timing_model_coefficient_mode="mean",
    timing_model_signal_name=None,
    residual_marker="o",
    residual_markersize=4.5,
    residual_marker_alpha=0.30,
    residual_error_alpha=0.08,
    residual_elinewidth=0.6,
    linewidth=2.2,
    interval_alpha=0.16,
    curve_alpha=0.055,
    ylim=None,
    figsize=(14, 5.5),
    ax=None,
):
    """
    Overlay either total polarization delay or distortion for multiple bands.

    For ``quantity="delay"``, this function plots the physical timing-delay
    sum,

        delta_t_pol(t)
            = Delta_X(t) b_X(t)
            + Delta_Y(t) b_Y(t)
            + Delta_Z(t) b_Z(t),

    For ``quantity="distortion"``, it instead plots the boost-vector
    magnitude,

        |b(t)| = sqrt(b_X(t)^2 + b_Y(t)^2 + b_Z(t)^2).

    Both derived quantities and their credible intervals are evaluated
    realization by realization. Each band is evaluated on its own observing
    span and dense time grid and is assigned a different color.

    When ``show_residuals=True`` and ``remove_timing_model=True``, the plotted
    TOA residuals are corrected using the posterior conditional mean (or one
    conditional sample, if requested) of the marginalized Enterprise linear
    timing-model process,

        r_plot = r_enterprise - delta_t_TM.

    This removes only the linear correction represented by
    ``gp_signals.TimingModel()``. It does not subtract the full TEMPO2 timing
    model a second time.

    Parameters
    ----------
    pta
        Enterprise PTA object containing the X, Y and Z polarization signals
        for every requested band.

    posterior_samples
        Posterior samples accepted by reconstruct_smooth_pol_gp().

    psr_name : str
        Pulsar name, for example ``"J1022+1001"``.

    bands : sequence of str
        Bands to overlay. The default is ``("10CM", "20CM", "40CM")``.

    quantity : {"delay", "distortion"}
        ``"delay"`` plots the projected total polarization timing delay in
        microseconds. ``"distortion"`` plots the dimensionless total
        polarization distortion magnitude ``|b(t)|``.

    band_colors : dict or None
        Optional mapping from band name to Matplotlib color. Missing entries
        are filled automatically.

    show_interval : bool
        Draw the pointwise credible interval for each band's total delay.

    show_sample_curves : bool
        Draw a limited number of faint posterior total-delay realizations.

    show_observed_gp : bool
        Mark the posterior-median total GP evaluated at the original TOA
        epochs. These are reconstructed GP values, not measured residuals.

    psr_object
        Unfiltered pulsar object containing all requested bands, with
        ``toas``, ``residuals``, ``toaerrs`` and ``flags`` attributes. It is
        required when ``show_residuals=True`` and must be the same unfiltered
        pulsar object used to build the PTA if timing-model removal is enabled.

    show_residuals : bool
        Plot measured TOA residuals and their uncertainties in the background.
        TOAs are selected by ``psr_object.flags[flagname]`` and colored using
        their observing band.

    remove_timing_model : bool
        If True, replace the raw ``psr_object.residuals`` points by
        timing-model-corrected residuals
        ``r - delta_t_TM``. This requires ``show_residuals=True``.

    timing_model_coefficient_mode : {"mean", "sample"}
        Conditional timing-model reconstruction mode. ``"mean"`` is
        recommended for residual plotting.

    timing_model_signal_name : str or None
        Optional explicit Enterprise linear timing-model signal name or suffix.

    ax : matplotlib.axes.Axes or None
        Existing axis on which to draw. A new figure is created when omitted.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    reconstructions : dict
        Mapping ``band -> reconstruct_pol_xyz_total(...)``. If linear timing
        removal is requested, every band result also contains the shared
        timing reconstruction under ``["timing_model"]``. The plotted
        sample-by-sample polarization totals remain available as
        ``reconstructions[band]["total"]["gp_us"]``.
    """

    bands = tuple(str(band) for band in bands)
    quantity = str(quantity).strip().lower()

    if quantity not in {"delay", "distortion"}:
        raise ValueError(
            "quantity must be either 'delay' or 'distortion'."
        )

    if not bands:
        raise ValueError("At least one band must be supplied.")

    if len(set(bands)) != len(bands):
        raise ValueError("Band names must be unique.")

    if not 0.0 < credible_level < 1.0:
        raise ValueError("credible_level must be between 0 and 1.")

    if interval_alpha < 0.0 or interval_alpha > 1.0:
        raise ValueError("interval_alpha must be between 0 and 1.")

    if curve_alpha < 0.0 or curve_alpha > 1.0:
        raise ValueError("curve_alpha must be between 0 and 1.")

    if remove_timing_model and not show_residuals:
        raise ValueError(
            "remove_timing_model=True requires show_residuals=True."
        )

    if quantity == "distortion" and show_residuals:
        raise ValueError(
            "show_residuals=True is only valid for quantity='delay'. "
            "Measured residuals are in microseconds, whereas the total "
            "polarization distortion is dimensionless."
        )

    if quantity == "distortion" and show_observed_gp:
        warnings.warn(
            "show_observed_gp is ignored for quantity='distortion'; "
            "it represents the projected GP at the TOAs, not |b(t)|.",
            RuntimeWarning,
        )
        show_observed_gp = False

    default_colors = {
        "10CM": "#0072B2",
        "20CM": "#E69F00",
        "40CM": "#009E73",
        "50CM": "#CC79A7",
    }

    if band_colors is None:
        band_colors = {}
    else:
        band_colors = dict(band_colors)

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get(
        "color",
        ["tab:blue", "tab:orange", "tab:green"],
    )

    for index, band in enumerate(bands):
        band_colors.setdefault(
            band,
            default_colors.get(
                band,
                color_cycle[index % len(color_cycle)],
            ),
        )

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    # --------------------------------------------------------
    # Reconstruct all requested bands first. Doing this before
    # the residual correction gives us the exact posterior sample
    # indices used by the polarization reconstruction.
    # --------------------------------------------------------

    reconstructions = {}

    for band in bands:
        print(
            f"Reconstructing total polarization {quantity} "
            f"for {band} ..."
        )

        reconstructions[band] = reconstruct_pol_xyz_total(
            pta=pta,
            posterior_samples=posterior_samples,
            psr_name=psr_name,
            band=band,
            N=N,
            n_grid=n_grid,
            selection=selection,
            seed=seed,
            flagname=flagname,
            coefficient_mode=coefficient_mode,
            phiinv_method=phiinv_method,
            max_gap_days=max_gap_days,
            validate=validate,
            credible_level=credible_level,
        )

    reference_indices = np.asarray(
        reconstructions[bands[0]]["total"]["selected_indices"]
    )

    for band in bands[1:]:
        current_indices = np.asarray(
            reconstructions[band]["total"]["selected_indices"]
        )
        if not np.array_equal(reference_indices, current_indices):
            raise RuntimeError(
                f"{band} used different posterior samples. The overlaid "
                "band reconstructions would not correspond to the same "
                "posterior draws."
            )

    # --------------------------------------------------------
    # Residual data. If requested, remove only the posterior
    # linear timing-model contribution returned by ConditionalGP.
    # --------------------------------------------------------

    timing_reconstruction = None

    if show_residuals:
        if psr_object is None:
            raise ValueError(
                "psr_object must be supplied when show_residuals=True."
            )

        for attribute in ("toas", "residuals", "toaerrs", "flags"):
            if not hasattr(psr_object, attribute):
                raise AttributeError(
                    f"psr_object has no {attribute!r} attribute."
                )

        if flagname not in psr_object.flags:
            raise KeyError(
                f"Flag {flagname!r} is not present in psr_object.flags. "
                f"Available flags: {list(psr_object.flags.keys())}"
            )

        residual_mjd = (
            np.asarray(psr_object.toas, dtype=float)
            / 86400.0
        )
        residual_error_us = (
            np.asarray(psr_object.toaerrs, dtype=float)
            * 1e6
        )
        residual_band = _normalise_flags(
            psr_object.flags[flagname]
        )

        if remove_timing_model:
            print(
                f"Reconstructing linear timing-model contribution for "
                f"{psr_name} ..."
            )

            timing_reconstruction = reconstruct_linear_timing_model(
                pta=pta,
                posterior_samples=posterior_samples,
                psr_object=psr_object,
                N=N,
                selection=selection,
                seed=seed,
                selected_indices=reference_indices,
                coefficient_mode=timing_model_coefficient_mode,
                phiinv_method=phiinv_method,
                credible_level=credible_level,
                timing_model_signal_name=timing_model_signal_name,
            )

            residual_us = np.asarray(
                timing_reconstruction["corrected_median_us"],
                dtype=float,
            )

            for band in bands:
                reconstructions[band]["timing_model"] = (
                    timing_reconstruction
                )
        else:
            residual_us = (
                np.asarray(psr_object.residuals, dtype=float)
                * 1e6
            )

        if not (
            residual_mjd.size
            == residual_us.size
            == residual_error_us.size
            == residual_band.size
        ):
            raise ValueError(
                "TOAs, residuals, TOA errors and band flags must have "
                "the same length."
            )

    interval_percent = 100.0 * credible_level

    # --------------------------------------------------------
    # Plot residuals and reconstructed polarization process.
    # --------------------------------------------------------

    for band in bands:
        color = band_colors[band]

        if show_residuals:
            residual_mask = (
                (residual_band == band)
                & np.isfinite(residual_mjd)
                & np.isfinite(residual_us)
                & np.isfinite(residual_error_us)
            )

            if np.any(residual_mask):
                order = np.argsort(residual_mjd[residual_mask])
                band_mjd = residual_mjd[residual_mask][order]
                band_residual_us = residual_us[residual_mask][order]
                band_error_us = residual_error_us[residual_mask][order]

                ax.errorbar(
                    band_mjd,
                    band_residual_us,
                    yerr=band_error_us,
                    fmt=residual_marker,
                    color=to_rgba(color, residual_marker_alpha),
                    ecolor=to_rgba(color, residual_error_alpha),
                    markersize=residual_markersize,
                    markeredgewidth=0.0,
                    elinewidth=residual_elinewidth,
                    capsize=0,
                    linestyle="none",
                    zorder=0,
                )

        reconstruction = reconstructions[band]
        mjd = np.asarray(reconstruction["mjd"], dtype=float)
        total = reconstruction["total"]

        if quantity == "delay":
            total_draws = np.asarray(total["gp_us"], dtype=float)
            total_lower = np.asarray(total["lower_us"], dtype=float)
            total_median = np.asarray(total["median_us"], dtype=float)
            total_upper = np.asarray(total["upper_us"], dtype=float)
        else:
            boost_x = np.asarray(
                reconstruction["X"]["boost_process"],
                dtype=float,
            )
            boost_y = np.asarray(
                reconstruction["Y"]["boost_process"],
                dtype=float,
            )
            boost_z = np.asarray(
                reconstruction["Z"]["boost_process"],
                dtype=float,
            )

            if not (boost_x.shape == boost_y.shape == boost_z.shape):
                raise RuntimeError(
                    f"X/Y/Z boost arrays have different shapes for {band}."
                )

            total_draws = np.sqrt(
                boost_x**2
                + boost_y**2
                + boost_z**2
            )

            alpha_q = 0.5 * (1.0 - credible_level)
            total_lower, total_median, total_upper = np.nanquantile(
                total_draws,
                [alpha_q, 0.5, 1.0 - alpha_q],
                axis=0,
            )

            reconstruction["total_distortion"] = {
                "process_name": (
                    f"{psr_name}_pol_cal_total_distortion_{band}"
                ),
                "boost_process": total_draws,
                "b_process": total_draws,
                "lower": total_lower,
                "median": total_median,
                "upper": total_upper,
                "selected_indices": np.asarray(
                    total["selected_indices"]
                ),
                "credible_level": credible_level,
            }

        if show_sample_curves and max_curves is not None and max_curves > 0:
            n_show = min(int(max_curves), total_draws.shape[0])
            curve_indices = np.unique(
                np.linspace(
                    0,
                    total_draws.shape[0] - 1,
                    n_show,
                    dtype=int,
                )
            )

            for curve_index in curve_indices:
                ax.plot(
                    mjd,
                    total_draws[curve_index],
                    color=color,
                    linewidth=0.65,
                    alpha=curve_alpha,
                    zorder=1,
                )

        if show_interval:
            ax.fill_between(
                mjd,
                total_lower,
                total_upper,
                color=color,
                alpha=interval_alpha,
                linewidth=0.0,
                zorder=2,
            )

        line_label = (
            f"{band}"
            if quantity == "delay"
            else f"{band} |b(t)|"
        )

        ax.plot(
            mjd,
            total_median,
            color=color,
            linewidth=linewidth,
            label=line_label,
            zorder=4,
        )

        if show_observed_gp:
            observed_mjd = np.asarray(
                reconstruction["observed_mjd"],
                dtype=float,
            )
            band_mask = np.asarray(
                reconstruction["band_mask"],
                dtype=bool,
            )
            observed_median = np.asarray(
                total["observed_median_us"],
                dtype=float,
            )

            ax.scatter(
                observed_mjd[band_mask],
                observed_median[band_mask],
                s=9,
                color=color,
                alpha=0.65,
                zorder=5,
            )

    ax.axhline(
        0.0,
        color="grey",
        linewidth=0.8,
        alpha=0.45,
        zorder=0,
    )
    ax.set_xlabel("MJD")
    if quantity == "delay":
        ax.set_ylabel(r"$\mathrm{PIB}^{-}$ delay ($\mu$s)")
    else:
        ax.set_ylabel(r"Total $\mathrm{PIB}^{-}$ $|\mathbf{b}(t)|$")
    ax.grid(True, color="0.88", linewidth=0.7, alpha=0.65)

    if ylim is not None:
        if np.isscalar(ylim):
            limit = abs(float(ylim))
            ax.set_ylim(-limit, limit)
        else:
            if len(ylim) != 2:
                raise ValueError(
                    "ylim must be a scalar, two-element sequence, or None."
                )
            ax.set_ylim(float(ylim[0]), float(ylim[1]))

    ax.legend(loc="upper right", ncol=(3 if show_residuals else 1))

    if ax.figure is fig:
        fig.tight_layout()

    return fig, ax, reconstructions


def plot_total_pol_distortion_by_band(*args, **kwargs):
    """
    Plot ``|b(t)|`` for all requested bands on one common axis.

    This is a convenience wrapper around plot_total_pol_delay_by_band() with
    ``quantity="distortion"``. All band colors, credible intervals, axis
    limits and figure options are controlled by the same keyword arguments.
    """

    if "quantity" in kwargs and kwargs["quantity"] != "distortion":
        raise ValueError(
            "plot_total_pol_distortion_by_band always uses "
            "quantity='distortion'."
        )

    kwargs["quantity"] = "distortion"
    return plot_total_pol_delay_by_band(*args, **kwargs)


def plot_total_pol_delay_sum_across_bands(
    pta,
    posterior_samples,
    psr_name,
    bands=("10CM", "20CM", "40CM"),
    *,
    psr_object=None,
    N=100,
    n_grid=5000,
    common_n_grid=None,
    selection="random",
    seed=1234,
    flagname="B",
    coefficient_mode="mean",
    phiinv_method="cliques",
    max_gap_days=None,
    validate=True,
    credible_level=0.90,
    total_color="black",
    band_colors=None,
    show_interval=True,
    show_band_delays=False,
    show_band_intervals=False,
    show_residuals=False,
    remove_timing_model=False,
    timing_model_coefficient_mode="mean",
    timing_model_signal_name=None,
    residual_marker="o",
    residual_markersize=4.5,
    residual_marker_alpha=0.30,
    residual_error_alpha=0.08,
    residual_elinewidth=0.6,
    total_linewidth=2.8,
    band_linewidth=1.4,
    interval_alpha=0.18,
    band_interval_alpha=0.08,
    show_noise_processes=False,
    noise_processes=("rn", "dm", "gw"),
    noise_process_colors=None,
    noise_process_styles=None,
    noise_process_name_overrides=None,
    show_noise_intervals=True,
    noise_interval_alpha=0.10,
    noise_linewidth=2.0,
    dm_frequency_mhz=1400.0,
    skip_missing_noise_processes=True,
    ylim=None,
    figsize=(14, 5.5),
    ax=None,
):
    """
    Plot the realization-by-realization sum of total delays across bands.

    Each band's total delay is first formed from its X, Y and Z projected
    polarization components. The requested band totals are then evaluated on
    their common overlapping MJD range and summed realization by realization,

        delta_t_all(t) = sum_band delta_t_band(t).

    Posterior quantiles are calculated only after this sum. This preserves
    correlations between axes and bands present in each posterior draw.

    When ``show_residuals=True`` and ``remove_timing_model=True``, the plotted
    residual points are corrected by subtracting the posterior conditional
    Enterprise linear timing-model contribution,

        r_plot = r_enterprise - delta_t_TM.

    The correction is reconstructed jointly with all basis processes through
    ``utils.ConditionalGP`` and therefore uses the same linear timing basis
    and covariance model as the Enterprise likelihood.

    Parameters
    ----------
    common_n_grid : int or None
        Number of points on the common overlapping time grid. When None,
        ``n_grid`` is used.

    total_color
        Matplotlib color for the summed delay.

    show_band_delays : bool
        Also draw the median total delay of every individual band.

    show_band_intervals : bool
        Also draw the credible interval of every individual band. This option
        is used only when ``show_band_delays=True``.

    show_residuals : bool
        Plot measured residuals and TOA uncertainties in the background,
        selected and colored by observing band. ``psr_object`` is required.

    remove_timing_model : bool
        If True, use ``r - delta_t_TM`` instead of raw
        ``psr_object.residuals`` for the plotted residual points. This requires
        ``show_residuals=True`` and the same unfiltered pulsar object used to
        build the PTA.

    timing_model_coefficient_mode : {"mean", "sample"}
        Conditional timing-model reconstruction mode. ``"mean"`` is
        recommended for residual plotting.

    timing_model_signal_name : str or None
        Optional explicit Enterprise linear timing-model signal name or suffix.

    show_noise_processes : bool
        Reconstruct and overlay the requested red-noise, DM and GWB timing
        processes on the same axis.

    noise_processes : sequence of {"rn", "dm", "gw"}
        Processes to request. Missing signals are skipped by default, allowing
        pulsars that contain only RN or only DM to be plotted normally.

    noise_process_name_overrides : dict or None
        Optional explicit signal names keyed by ``"rn"``, ``"dm"`` or
        ``"gw"`` when the PTA uses non-standard names.

    dm_frequency_mhz : float
        Frequency at which the chromatic DM delay is displayed. The default
        is 1400 MHz.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    result : dict
        ``result["gp_us"]`` contains the sample-by-sample cross-band sum.
        ``result["median_us"]``, ``result["lower_us"]`` and
        ``result["upper_us"]`` contain its pointwise summaries. Individual
        reconstructions are stored under ``result["reconstructions"][band]``
        and their interpolated total-delay draws under
        ``result["band_gp_us"][band]``. If requested,
        ``result["timing_model"]`` contains the linear timing-model
        reconstruction and timing-model-corrected residuals.
    """

    bands = tuple(str(band) for band in bands)

    if not bands:
        raise ValueError("At least one band must be supplied.")

    if len(set(bands)) != len(bands):
        raise ValueError("Band names must be unique.")

    if not 0.0 < credible_level < 1.0:
        raise ValueError("credible_level must be between 0 and 1.")

    if remove_timing_model and not show_residuals:
        raise ValueError(
            "remove_timing_model=True requires show_residuals=True."
        )

    common_n_grid = n_grid if common_n_grid is None else int(common_n_grid)
    if common_n_grid < 2:
        raise ValueError("common_n_grid must be at least 2.")

    default_colors = {
        "10CM": "#0072B2",
        "20CM": "#E69F00",
        "40CM": "#009E73",
        "50CM": "#CC79A7",
    }

    band_colors = {} if band_colors is None else dict(band_colors)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get(
        "color",
        ["tab:blue", "tab:orange", "tab:green"],
    )

    for index, band in enumerate(bands):
        band_colors.setdefault(
            band,
            default_colors.get(
                band,
                color_cycle[index % len(color_cycle)],
            ),
        )

    reconstructions = {}

    for band in bands:
        print(f"Reconstructing total polarization delay for {band} ...")
        reconstructions[band] = reconstruct_pol_xyz_total(
            pta=pta,
            posterior_samples=posterior_samples,
            psr_name=psr_name,
            band=band,
            N=N,
            n_grid=n_grid,
            selection=selection,
            seed=seed,
            flagname=flagname,
            coefficient_mode=coefficient_mode,
            phiinv_method=phiinv_method,
            max_gap_days=max_gap_days,
            validate=validate,
            credible_level=credible_level,
        )

    reference_indices = np.asarray(
        reconstructions[bands[0]]["total"]["selected_indices"]
    )

    for band in bands[1:]:
        current_indices = np.asarray(
            reconstructions[band]["total"]["selected_indices"]
        )
        if not np.array_equal(reference_indices, current_indices):
            raise RuntimeError(
                f"{band} used different posterior samples. The cross-band "
                "sum would not be a valid realization-by-realization sum."
            )

    overlap_start = max(
        float(np.nanmin(reconstructions[band]["mjd"]))
        for band in bands
    )
    overlap_end = min(
        float(np.nanmax(reconstructions[band]["mjd"]))
        for band in bands
    )

    if not overlap_end > overlap_start:
        raise ValueError(
            "The requested bands do not share an overlapping MJD range."
        )

    common_mjd = np.linspace(
        overlap_start,
        overlap_end,
        common_n_grid,
    )

    noise_reconstructions = {}

    if show_noise_processes:
        if psr_object is None:
            raise ValueError(
                "psr_object must be supplied when "
                "show_noise_processes=True. Pass the unfiltered pulsar "
                "object used to construct the PTA."
            )

        for requested_kind in noise_processes:
            canonical_kind = {
                "rn": "rn",
                "red": "rn",
                "red_noise": "rn",
                "dm": "dm",
                "dm_gp": "dm",
                "gw": "gw",
                "gwb": "gw",
                "gwb_hd": "gw",
            }.get(str(requested_kind).strip().lower())

            if canonical_kind is None:
                raise ValueError(
                    f"Unknown noise process {requested_kind!r}. "
                    "Use 'rn', 'dm', or 'gw'."
                )

            if canonical_kind in noise_reconstructions:
                continue

            process_name = _resolve_standard_process_name(
                pta=pta,
                psr_name=psr_name,
                process_kind=canonical_kind,
                process_name_overrides=noise_process_name_overrides,
            )

            if process_name is None:
                message = (
                    f"No {canonical_kind.upper()} signal found for "
                    f"{psr_name}."
                )
                if skip_missing_noise_processes:
                    print(message + " Skipping it.")
                    continue
                raise KeyError(message)

            print(f"Reconstructing {process_name} ...")
            process_reconstruction = reconstruct_smooth_standard_gp(
                pta=pta,
                posterior_samples=posterior_samples,
                process_name=process_name,
                psr_object=psr_object,
                process_kind=canonical_kind,
                N=N,
                grid_mjd=common_mjd,
                selection=selection,
                seed=seed,
                coefficient_mode=coefficient_mode,
                phiinv_method=phiinv_method,
                credible_level=credible_level,
                dm_frequency_mhz=dm_frequency_mhz,
                validate=validate,
            )

            process_indices = np.asarray(
                process_reconstruction["selected_indices"]
            )
            if not np.array_equal(reference_indices, process_indices):
                raise RuntimeError(
                    f"{process_name} used different posterior samples from "
                    "the polarization reconstruction."
                )

            noise_reconstructions[canonical_kind] = (
                process_reconstruction
            )

    def _interpolate_draws_preserving_gaps(source_mjd, draws):
        """Interpolate each draw without bridging explicit NaN gaps."""

        source_mjd = np.asarray(source_mjd, dtype=float)
        draws = np.asarray(draws, dtype=float)

        if draws.ndim != 2 or draws.shape[1] != source_mjd.size:
            raise ValueError(
                "Each band GP must have shape (n_draws, n_mjd)."
            )

        interpolated = np.full(
            (draws.shape[0], common_mjd.size),
            np.nan,
            dtype=float,
        )

        for draw_index, draw in enumerate(draws):
            finite_indices = np.flatnonzero(
                np.isfinite(source_mjd) & np.isfinite(draw)
            )

            if finite_indices.size < 2:
                continue

            split_locations = np.flatnonzero(
                np.diff(finite_indices) > 1
            ) + 1
            finite_runs = np.split(finite_indices, split_locations)

            for run in finite_runs:
                if run.size < 2:
                    continue

                run_mjd = source_mjd[run]
                run_values = draw[run]
                target_mask = (
                    (common_mjd >= run_mjd[0])
                    & (common_mjd <= run_mjd[-1])
                )

                if np.any(target_mask):
                    interpolated[draw_index, target_mask] = (
                        PchipInterpolator(
                            run_mjd,
                            run_values,
                            extrapolate=False,
                        )(common_mjd[target_mask])
                    )

        return interpolated

    band_gp_us = {}

    for band in bands:
        band_gp_us[band] = _interpolate_draws_preserving_gaps(
            reconstructions[band]["mjd"],
            reconstructions[band]["total"]["gp_us"],
        )

    band_shapes = {draws.shape for draws in band_gp_us.values()}
    if len(band_shapes) != 1:
        raise RuntimeError(
            "Interpolated band-delay arrays have different shapes."
        )

    stacked_band_draws = np.stack(
        [band_gp_us[band] for band in bands],
        axis=0,
    )

    # Ordinary sum is deliberate: if any requested band is undefined inside
    # a preserved data gap, the combined process is also undefined there.
    summed_gp_us = np.sum(stacked_band_draws, axis=0)

    alpha_q = 0.5 * (1.0 - credible_level)
    quantiles = [alpha_q, 0.5, 1.0 - alpha_q]
    summed_lower, summed_median, summed_upper = np.nanquantile(
        summed_gp_us,
        quantiles,
        axis=0,
    )

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    # --------------------------------------------------------
    # Measured residuals. The timing-model correction is
    # reconstructed once using the exact same posterior draws as
    # the polarization and optional noise reconstructions.
    # --------------------------------------------------------

    timing_reconstruction = None

    if show_residuals:
        if psr_object is None:
            raise ValueError(
                "psr_object must be supplied when show_residuals=True."
            )

        for attribute in ("toas", "residuals", "toaerrs", "flags"):
            if not hasattr(psr_object, attribute):
                raise AttributeError(
                    f"psr_object has no {attribute!r} attribute."
                )

        if flagname not in psr_object.flags:
            raise KeyError(
                f"Flag {flagname!r} is not present in psr_object.flags."
            )

        residual_mjd = np.asarray(
            psr_object.toas,
            dtype=float,
        ) / 86400.0
        residual_error_us = (
            np.asarray(psr_object.toaerrs, dtype=float)
            * 1e6
        )
        residual_band = _normalise_flags(
            psr_object.flags[flagname]
        )

        if remove_timing_model:
            print(
                f"Reconstructing linear timing-model contribution for "
                f"{psr_name} ..."
            )

            timing_reconstruction = reconstruct_linear_timing_model(
                pta=pta,
                posterior_samples=posterior_samples,
                psr_object=psr_object,
                N=N,
                selection=selection,
                seed=seed,
                selected_indices=reference_indices,
                coefficient_mode=timing_model_coefficient_mode,
                phiinv_method=phiinv_method,
                credible_level=credible_level,
                timing_model_signal_name=timing_model_signal_name,
            )

            residual_us = np.asarray(
                timing_reconstruction["corrected_median_us"],
                dtype=float,
            )
        else:
            residual_us = (
                np.asarray(psr_object.residuals, dtype=float)
                * 1e6
            )

        if not (
            residual_mjd.size
            == residual_us.size
            == residual_error_us.size
            == residual_band.size
        ):
            raise ValueError(
                "TOAs, residuals, TOA errors and band flags must have "
                "the same length."
            )

        for band in bands:
            residual_mask = (
                (residual_band == band)
                & np.isfinite(residual_mjd)
                & np.isfinite(residual_us)
                & np.isfinite(residual_error_us)
            )

            if not np.any(residual_mask):
                continue

            order = np.argsort(residual_mjd[residual_mask])
            ax.errorbar(
                residual_mjd[residual_mask][order],
                residual_us[residual_mask][order],
                yerr=residual_error_us[residual_mask][order],
                fmt=residual_marker,
                color=to_rgba(
                    band_colors[band],
                    residual_marker_alpha,
                ),
                ecolor=to_rgba(
                    band_colors[band],
                    residual_error_alpha,
                ),
                markersize=residual_markersize,
                markeredgewidth=0.0,
                elinewidth=residual_elinewidth,
                capsize=0,
                linestyle="none",
                label=f"{band}",
                zorder=0,
            )

    default_noise_colors = {
        "rn": "#D55E00",
        "dm": "#CC79A7",
        "gw": "#56B4E9",
    }
    default_noise_styles = {
        "rn": "--",
        "dm": "-.",
        "gw": ":",
    }
    noise_process_colors = (
        {}
        if noise_process_colors is None
        else dict(noise_process_colors)
    )
    noise_process_styles = (
        {}
        if noise_process_styles is None
        else dict(noise_process_styles)
    )

    for kind, color in default_noise_colors.items():
        noise_process_colors.setdefault(kind, color)
    for kind, style in default_noise_styles.items():
        noise_process_styles.setdefault(kind, style)

    if show_band_delays:
        for band in bands:
            band_lower, band_median, band_upper = np.nanquantile(
                band_gp_us[band],
                quantiles,
                axis=0,
            )

            if show_band_intervals:
                ax.fill_between(
                    common_mjd,
                    band_lower,
                    band_upper,
                    color=band_colors[band],
                    alpha=band_interval_alpha,
                    linewidth=0.0,
                    zorder=1,
                )

            ax.plot(
                common_mjd,
                band_median,
                color=band_colors[band],
                linewidth=band_linewidth,
                alpha=0.8,
                label=f"{band}",
                zorder=2,
            )

    interval_percent = 100.0 * credible_level

    if show_interval:
        ax.fill_between(
            common_mjd,
            summed_lower,
            summed_upper,
            color=total_color,
            alpha=interval_alpha,
            linewidth=0.0,
            zorder=3,
        )

    total_label = r"$\mathrm{PIB}^{-}$"

    ax.plot(
        common_mjd,
        summed_median,
        color=total_color,
        linewidth=total_linewidth,
        label=total_label,
        zorder=4,
    )

    process_labels = {
        "rn": "Red noise",
        "dm": f"DM GP ({float(dm_frequency_mhz):g} MHz)",
        "gw": "GWB (HD)",
    }

    for kind in ("rn", "dm", "gw"):
        if kind not in noise_reconstructions:
            continue

        process = noise_reconstructions[kind]
        color = noise_process_colors[kind]

        if show_noise_intervals:
            ax.fill_between(
                common_mjd,
                process["lower_us"],
                process["upper_us"],
                color=color,
                alpha=noise_interval_alpha,
                linewidth=0.0,
                zorder=3,
            )

        label = process_labels[kind]

        ax.plot(
            common_mjd,
            process["median_us"],
            color=color,
            linestyle=noise_process_styles[kind],
            linewidth=noise_linewidth,
            label=label,
            zorder=5,
        )

    ax.axhline(
        0.0,
        color="grey",
        linewidth=0.8,
        alpha=0.45,
        zorder=0,
    )
    ax.set_xlabel("MJD")
    ax.set_ylabel(r"Residuals ($\mu$s)")
    ax.grid(True, color="0.88", linewidth=0.7, alpha=0.65)

    if ylim is not None:
        if np.isscalar(ylim):
            limit = abs(float(ylim))
            ax.set_ylim(-limit, limit)
        else:
            if len(ylim) != 2:
                raise ValueError(
                    "ylim must be a scalar, two-element sequence, or None."
                )
            ax.set_ylim(float(ylim[0]), float(ylim[1]))

    ax.legend(loc="best", ncol=3)
    fig.tight_layout()

    result = {
        "mjd": common_mjd,
        "gp_us": summed_gp_us,
        "lower_us": summed_lower,
        "median_us": summed_median,
        "upper_us": summed_upper,
        "bands": bands,
        "band_gp_us": band_gp_us,
        "reconstructions": reconstructions,
        "noise_processes": noise_reconstructions,
        "timing_model": timing_reconstruction,
        "selected_indices": reference_indices,
        "credible_level": credible_level,
    }

    return fig, ax, result


def mask_all_components(psr, flag_key='B', desired_value=True, in_place=False):
    if not in_place:
        psr = copy.deepcopy(psr)
    
    flag_array = np.array(psr.flags[flag_key])
    mask = flag_array == desired_value
    mask_length = len(mask)
    new_length = int(np.sum(mask))

    def try_mask(val):
        if isinstance(val, (str, bytes)):
            return val
        try:
            if hasattr(val, '__len__') and len(val) == mask_length:
                return np.array(val)[mask]
        except Exception:
            pass
        return val

    for attr, value in psr.__dict__.items():
        #print(f"Processing attribute: {attr}")
        # Overwrite "isort" and "iisort" with a new index array
        if attr in ['_isort', '_iisort']:
            setattr(psr, attr, np.arange(new_length))
        elif isinstance(value, dict):
            new_dict = {}
            for key, val in value.items():
                new_dict[key] = try_mask(val)
            setattr(psr, attr, new_dict)
        else:
            setattr(psr, attr, try_mask(value))
    
    return psr

# filtered_psrs_10 = [mask_all_components(psr, flag_key='B', desired_value='10CM') for psr in psrs]
#filtered_psrs_by_band = {
#     "10CM": filtered_psrs_10,
#     "20CM": filtered_psrs_20,
#     "40CM": filtered_psrs_40,
# }
