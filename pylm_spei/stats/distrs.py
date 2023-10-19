import numpy as np
import numba as nb
from numba_stats import norm



def genextreme_cdf(x:np.ndarray,c:np.ndarray,loc:np.ndarray,scale:np.ndarray) -> np.ndarray:
    z = z = 1 - c * (x - loc) / scale
    return np.exp(-z ** (-1 / c))

def norm_ppf(p:np.ndarray,loc:np.ndarray,scale:np.ndarray) -> np.ndarray:
    return norm.ppf(p,loc=loc,scale=scale)