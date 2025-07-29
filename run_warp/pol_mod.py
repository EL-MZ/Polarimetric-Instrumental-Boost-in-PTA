import logging
import itertools
import numpy as np

from enterprise.signals import selections, utils
from enterprise.signals.selections import Selection

from enterprise.signals import parameter, selections, signal_base, utils
from enterprise.signals.selections import Selection
from enterprise.signals.parameter import function

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
    
    basis = createfourierdesignmatrix_red_pol_diag(
        pol_axis=pol_axis,nmodes=components, Tspan=Tspan, logf=logf, fmin=fmin, fmax=fmax, modes=modes, pshift=pshift, pseed=pseed
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
