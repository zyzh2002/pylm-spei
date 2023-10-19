from .do_calcs.calc_gev_paras_lm import get_gev_lm_paras
from .do_calcs.calc_lmoments import get_lratios_jit
from numpy import ndarray
from .stats.distrs import norm_ppf,genextreme_cdf


def calc_spei(d_i_mat:ndarray)-> ndarray:
    """
    Calculate SPEI from a 3D array of precipitation data.
    """
    # Calculate L-moments
    lm_est = get_lratios_jit(d_i_mat, nmom=3)
    # Calculate GEV parameters
    gev_paras = get_gev_lm_paras(lm_est)
    # Calculate probability
    p_i_mat = genextreme_cdf(d_i_mat, c=gev_paras[0], loc=gev_paras[1], scale=gev_paras[2])
    # Standardize probability and get SPEI
    spei = norm_ppf(p_i_mat, loc=0, scale=1)
    return spei
