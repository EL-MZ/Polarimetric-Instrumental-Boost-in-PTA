import logging
import itertools
import functools
import numpy as np


from enterprise.signals import parameter, selections, signal_base
from enterprise.signals.selections import Selection
from enterprise.signals.parameter import function
from enterprise.signals.utils import KernelMatrix
import enterprise.signals.utils as utils

# logging.basicConfig(format="%(levelname)s: %(name)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)



def createfourierdesignmatrix_red_pol_diag_old(
    toas,distort_vect,pol_axis = "X", nmodes=30, Tspan=None, logf=False, fmin=None, fmax=None, pshift=False, modes=None, pseed=None
):
    """
    Construct fourier design matrix from eq 11 of Lentati et al, 2013 old
    Constructing a fourier design matrix for a Polarization distortion model. 
    :param dist_vec will be the attribute of the 3D vector of the pulsar's distortion vector.
    :param toas: vector of time series in seconds
    :param nmodes: number of fourier coefficients to use
    :param freq: option to output frequencies
    :param Tspan: option to some other Tspan
    :param logf: use log frequency spacing
    :param fmin: lower sampling frequency
    :param fmax: upper sampling frequency
    :param pshift: option to add random phase shift
    :param pseed: option to provide phase shift seed
    :param modes: option to provide explicit list or array of
                  sampling frequencies

    :return: F: fourier design matrix
    :return: f: Sampling frequencies
    """

    T = Tspan if Tspan is not None else toas.max() - toas.min()

    # define sampling frequencies
    if modes is not None:
        nmodes = len(modes)
        f = modes
    elif fmin is None and fmax is None and not logf:
        # make sure partially overlapping sets of modes
        # have identical frequencies
        f = 1.0 * np.arange(1, nmodes + 1) / T
    else:
        # more general case

        if fmin is None:
            fmin = 1 / T

        if fmax is None:
            fmax = nmodes / T

        if logf:
            f = np.logspace(np.log10(fmin), np.log10(fmax), nmodes)
        else:
            f = np.linspace(fmin, fmax, nmodes)

    # if requested, add random phase shift to basis functions
    if pshift or pseed is not None:
        if pseed is not None:
            # use the first toa to make a different seed for every pulsar
            seed = int(toas[0] / 17) + int(pseed)
            np.random.seed(seed)

        ranphase = np.random.uniform(0.0, 2 * np.pi, nmodes)
    else:
        ranphase = np.zeros(nmodes)

    Ffreqs = np.repeat(f, 2)
    if pol_axis == "X":
        dist_vect =  distort_vect[:,0]
    elif pol_axis == "Y":
        dist_vect =  distort_vect[:,1]
    elif pol_axis == "Z":
        dist_vect =  distort_vect[:,2]
    else:
        dist_vect = np.ones(len(psr.toas)) 
        raise Warning("pol_axis must be 'X', 'Y', or 'Z' set to 1.")
         
    N = len(toas)
    F = np.zeros((N, 2 * nmodes))

    # The sine/cosine modes
    F[:, ::2] = dist_vect*np.sin(2 * np.pi * toas[:, None] * f[None, :] + ranphase[None, :])
    F[:, 1::2] = dist_vect*np.cos(2 * np.pi * toas[:, None] * f[None, :] + ranphase[None, :])

    return F, Ffreqs

@function
def createfourierdesignmatrix_red_pol_diag(
    toas,                       # injected automatically
    distort_vect,               # injected automatically
    pol_axis="X",
    nmodes=30,
    Tspan=None,
    logf=False,
    fmin=None,
    fmax=None,
    pshift=False,
    modes=None,
    pseed=None,
):
    
    axis = dict(X=0, Y=1, Z=2).get(pol_axis.upper())
    #print(f"Using axis {axis} for pol_axis {pol_axis}")
    if axis is None:
        raise ValueError("pol_axis must be 'X', 'Y' or 'Z'")

    T = Tspan if Tspan is not None else toas.max() - toas.min()

    # --- frequencies --------------------------------------------------------
    if modes is not None:
        f = np.asarray(modes, float)
    elif fmin is None and fmax is None and not logf:
        f = np.arange(1, nmodes + 1, dtype=float) / T
    else:
        fmin = 1.0 / T if fmin is None else fmin
        fmax = nmodes / T if fmax is None else fmax
        f = (np.logspace(np.log10(fmin), np.log10(fmax), nmodes)
             if logf else np.linspace(fmin, fmax, nmodes))

    # --- optional random phase ---------------------------------------------
    if pshift or pseed is not None:
        if pseed is not None:
            seed = int(toas[0] / 17) + int(pseed)
            np.random.seed(seed)
        phase = np.random.uniform(0.0, 2*np.pi, len(f))
    else:
        phase = np.zeros(len(f))

    use = distort_vect[:, axis]                # select X/Y/Z component
    N   = len(toas)
    F   = np.zeros((N, 2*len(f)))
    Ffreqs = np.repeat(f, 2)
    F[:, ::2]  = use[:, None] * np.sin(2*np.pi*toas[:, None]*f + phase)
    F[:, 1::2] = use[:, None] * np.cos(2*np.pi*toas[:, None]*f + phase)
    # F[:, ::2]  = np.sin(2*np.pi*toas[:, None]*f + phase)
    # F[:, 1::2] = np.cos(2*np.pi*toas[:, None]*f + phase)
    #print(f"Distort basis: {F[0,0]}")
    
    return F, Ffreqs


@function
def createfourierdesignmatrix_red_pol_diag_selec(
    toas,
    flags,
    distort_vect,               # injected automatically
    pol_axis="X",
    flagname="B",
    flagval=None,
    nmodes=30,
    Tspan=None,
    psrTspan=True,
    logf=False,
    fmin=None,
    fmax=None,
    modes=None,
    pshift=None,
    pseed=None,
):
    """
    Construct fourier design matrix with possibility of adding selection and/or chromatic index envelope.

    :param toas: vector of time series in seconds
    :param freqs: radio frequencies of observations [MHz]
    :param flags: Flags from timfiles
    :param nmodes: number of fourier coefficients to use
    :param Tspan: option to some other Tspan
    :param psrTspan: option to use pulsar time span. Used only if sub-group of ToAs is chosen
    :param logf: use log frequency spacing
    :param fmin: lower sampling frequency
    :param fmax: upper sampling frequency
    :param log10_Amp: log10 of the Amplitude [s]
    :param idx: Index of chromatic effects
    :param modes: option to provide explicit list or array of
                  sampling frequencies

    :return: F: fourier design matrix
    :return: f: Sampling frequencies
    """
    if flagval and not psrTspan:
        sel_toas = toas[np.where(flags[flagname] == flagval)]
        Tspan = sel_toas.max() - sel_toas.min()

    # get base fourier design matrix and frequencies
    F, Ffreqs = createfourierdesignmatrix_red_pol_diag(
        toas,distort_vect=distort_vect,pol_axis=pol_axis, nmodes=nmodes, Tspan=Tspan, logf=logf, fmin=fmin, fmax=fmax, modes=modes, pshift=pshift, pseed=pseed
    )



    # compute the mask for the selection
    if flagval:
        F *= np.array([flags[flagname] == flagval] * F.shape[1]).T
# print(f"for {pol_axis} F Distorted basis: {F}")
    return F, Ffreqs

def BasisCommonGP(priorFunction, basisFunction, orfFunction, coefficients=False, combine=True, name=""):
    class BasisCommonGP(signal_base.CommonSignal):
        signal_type = "common basis"
        signal_name = "common"
        signal_id = name

        basis_combine = combine

        _orf = orfFunction(name)
        _prior = priorFunction(name)

        def __init__(self, psr):
            super(BasisCommonGP, self).__init__(psr)
            self.name = self.psrname + "_" + self.signal_id

            pname = "_".join([psr.name, name])
            self._bases = basisFunction(pname, psr=psr)

            self._params, self._coefficients = {}, {}

            for par in itertools.chain(
                self._prior._params.values(), self._orf._params.values(), self._bases._params.values()
            ):
                self._params[par.name] = par

            self._psrpos = psr.pos

            if coefficients:
                self._construct_basis()

                # if we're given an instantiated coefficient vector
                # that's what we will use
                if isinstance(coefficients, parameter.Parameter):
                    self._coefficients[""] = coefficients
                    self._params[coefficients.name] = coefficients

                    return

                chain = itertools.chain(
                    self._prior._params.values(), self._orf._params.values(), self._bases._params.values()
                )
                priorargs = {par.name: self._params[par.name] for par in chain}

                logprior = parameter.Function(self._get_coefficient_logprior, **priorargs)

                size = self._basis.shape[1]

                cpar = parameter.GPCoefficients(logprior=logprior, size=size)(pname + "_coefficients")

                self._coefficients[""] = cpar
                self._params[cpar.name] = cpar

        @property
        def basis_params(self):
            """Get any varying basis parameters."""
            return [pp.name for pp in self._bases.params]

        # since this function has side-effects, it can only be cached
        # with limit=1, so it will run again if called with params different
        # than the last time
        @signal_base.cache_call("basis_params", limit=1)
        def _construct_basis(self, params={}):
            self._basis, self._labels = self._bases(params=params)

        if coefficients:

            def _get_coefficient_logprior(self, c, **params):
                # MV: for correlated GPs, the prior needs to use
                #     the coefficients for all GPs together;
                #     this may require parameter groups

                raise NotImplementedError("Need to implement common prior " + "for BasisCommonGP coefficients")

            @property
            def delay_params(self):
                return [pp.name for pp in self.params if "_coefficients" in pp.name]

            @signal_base.cache_call(["basis_params", "delay_params"])
            def get_delay(self, params={}):
                self._construct_basis(params)

                p = self._coefficients[""]
                c = params[p.name] if p.name in params else p.value
                return np.dot(self._basis, c)

            def get_basis(self, params={}):
                return None

            def get_phi(self, params):
                return None

            def get_phicross(cls, signal1, signal2, params):
                return None

            def get_phiinv(self, params):
                return None

        else:

            @property
            def delay_params(self):
                return []

            def get_delay(self, params={}):
                return 0

            def get_basis(self, params={}):
                self._construct_basis(params)

                return self._basis

            def get_phi(self, params):
                self._construct_basis(params)

                prior = BasisCommonGP._prior(self._labels, params=params)
                orf = BasisCommonGP._orf(self._psrpos, self._psrpos, params=params)

                return prior * orf

            @classmethod
            def get_phicross(cls, signal1, signal2, params):
                prior = BasisCommonGP._prior(signal1._labels, params=params)
                orf = BasisCommonGP._orf(signal1._psrpos, signal2._psrpos, params=params)

                return prior * orf

    return BasisCommonGP



def FourierBasisCommonGP_pol(
    spectrum,
    orf,
    flagname="B",
    flagval=None,
    pol_axis="X",
    coefficients=False,
    combine=True,
    components=20,
    Tspan=None,
    logf=False,
    fmin=None,
    fmax=None,
    modes=None,
    name="common_fourier",
    pshift=False,
    pseed=None,
):

    if coefficients and Tspan is None:
        raise ValueError(
            "With coefficients=True, FourierBasisCommonGP " + "requires that you specify Tspan explicitly."
        )
    
    basis = createfourierdesignmatrix_red_pol_diag_selec(
        pol_axis=pol_axis,flagname=flagname,flagval=flagval,nmodes=components, Tspan=Tspan, logf=logf, fmin=fmin, fmax=fmax, modes=modes, pshift=pshift, pseed=pseed
    )
    BaseClass = BasisCommonGP(spectrum, basis, orf, coefficients=coefficients, combine=combine, name=name)

    class FourierBasisCommonGP_pol(BaseClass):
        signal_type = "common basis"
        signal_name = "common red noise"
        signal_id = name

        _Tmin, _Tmax = [], []

        def __init__(self, psr):
            super(FourierBasisCommonGP_pol, self).__init__(psr)

            if Tspan is None:
                FourierBasisCommonGP_pol._Tmin.append(psr.toas.min())
                FourierBasisCommonGP_pol._Tmax.append(psr.toas.max())

        # since this function has side-effects, it can only be cached
        # with limit=1, so it will run again if called with params different
        # than the last time
        @signal_base.cache_call("basis_params", 1)
        def _construct_basis(self, params={}):
            span = Tspan if Tspan is not None else max(FourierBasisCommonGP_pol._Tmax) - min(FourierBasisCommonGP_pol._Tmin)
            self._basis, self._labels = self._bases(params=params, Tspan=span)

    return FourierBasisCommonGP_pol




def create_polarization_signals(
    flags,
    nfreqs=None,
    Tspan=None,
    flagname="B",
    log_A_range=(-20, -6),
    gamma_range=(0, 7),
):
    """
    Creates a dictionary of polarization signal models for a list of flages.

    For each flag in the list, this function generates 'x', 'y', and 'z'
    axis components, each with its own amplitude and spectral index parameters.
    It then combines these into a total signal model.

    :param flages: A list of strings to use as flages for parameter and signal names.
    :param nfreqs: Number of Fourier components for the signal model.
    :param Tspan: The time span for the Fourier basis.
    :param log_A_range: A tuple for the Uniform prior range of log10_A.
    :param gamma_range: A tuple for the Uniform prior range of gamma.

    :return: A dictionary where keys are 'pol_tot_{flag}' and values are the
             corresponding combined signal objects.
    """
    # This dictionary will store the final combined signals
    total_signals = []
    orf = utils.monopole_orf()
    # Loop through each provided flag (e.g., 'sys1', 'sys2')
    for flag in flags:
        axes = []
        # For each flag, create models for X, Y, and Z axes
        for axis in ["X", "Y", "Z"]:
            axis_lower = axis.lower()

            # 1. Dynamically create parameter names with the flag
            log10_A_name = f"log10_A_pol_{axis_lower}_{flag}"
            gamma_name = f"gamma_pol_{axis_lower}_{flag}"
            signal_name = f"pol_cal_{axis_lower}_{flag}"

            # 2. Define the prior parameters for amplitude and spectral index
            log10_A = parameter.Uniform(*log_A_range)(log10_A_name)
            gamma = parameter.Uniform(*gamma_range)(gamma_name)

            # 3. Create the power-law spectrum model
            powerlaw_spectrum = utils.powerlaw(log10_A=log10_A, gamma=gamma)

            # 4. Create the Fourier basis signal for the current axis
            pol_component = FourierBasisCommonGP_pol(
                powerlaw_spectrum,
                orf=orf,
                flagname=flagname,
                flagval=flag,
                pol_axis=axis,
                components=nfreqs,
                Tspan=Tspan,
                name=signal_name,
            )
            axes.append(pol_component)
        if axes:
            axes_signal = axes[0]
            for component in axes[1:]:
                axes_signal += component
            total_signals.append(axes_signal)

        

    if not total_signals:
    # Return a neutral value if no suffixes were provided.
        print("No pol signals generated")
        return None

    # Initialize the final signal with the first element of the list
    final_signal = total_signals[0]
    
    # Loop through the rest of the signals and add them to the final signal
    for signal in total_signals[1:]:
        final_signal += signal

    return final_signal

