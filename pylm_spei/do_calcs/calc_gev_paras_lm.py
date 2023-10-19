import numpy as np
import numba as nb

@nb.njit(nb.float64[:, :, :](nb.float64[:, :, :]),parallel=True,cache=True)
def get_gev_lm_paras(lm_est:np.ndarray) -> np.ndarray:
    kappa = (0.488138*lm_est[2]**1.70839)-(1.7631*lm_est[2]**0.981824)+0.285706
    alpha = (1.023602813*lm_est[2]**1.8850974-2.95087636*lm_est[2]**1.195591244+1.759614982)*lm_est[1]
    zeta = (-0.0937*lm_est[2]**4-0.2198*lm_est[2]**3+1.407*lm_est[2]**2-1.4825*lm_est[2]-0.6205)*lm_est[1]+lm_est[0]
    return np.stack((kappa,zeta,alpha))