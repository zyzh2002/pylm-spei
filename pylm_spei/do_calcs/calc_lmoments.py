import numba as nb
import numpy as np
import scipy as sp


# Calc l_ratios in this function
@nb.njit(nb.float64[:, :, :](nb.float64[:, :, :], nb.int8), parallel=True, cache=True)
def get_lratios_jit(x: np.ndarray, nmom: int = 5) -> np.ndarray:

    def sort_array_with_axis(x: np.ndarray, axis: int) -> np.ndarray:
        for i in nb.prange(x.shape[axis]):
            x[i] = np.sort(x[i])
        return x

    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]

    x = sort_array_with_axis(x, 0)

    sum_xtrans = np.empty_like(x[0])

    def comb(n, r):
        if r < 0 or r > n:
            return 0
        if r == 0 or r == n:
            return 1
        c = 1
        for i in range(1, r + 1):
            c = c * (n - i + 1) // i
        return c

    # First L-moment

    l1 = np.sum(x, axis=0) / comb(n, 1)

    if nmom == 1:
        return np.expand_dims(l1, axis=0)

    # Second L-moment

    comb1 = np.arange(n)
    coefl2 = 0.5 / comb(n, 2)
    sum_xtrans[:] = 0
    for i in nb.prange(n):
        sum_xtrans += (comb1[i] - comb1[n - 1 - i]) * x[i]
    l2 = coefl2 * sum_xtrans

    if nmom == 2:
        return np.stack((l1, l2))

    # Third L-moment

    comb3 = np.zeros(n)
    for i in range(n):
        comb3[i] = comb(i, 2)
    coefl3 = 1.0 / 3.0 / comb(n, 3)
    sum_xtrans[:] = 0
    for i in nb.prange(n):
        sum_xtrans += (
            comb3[i] - 2 * comb1[i] * comb1[n - 1 - i] + comb3[n - 1 - i]
        ) * x[i]
    l3 = coefl3 * sum_xtrans / l2

    if nmom == 3:
        return np.stack((l1, l2, l3))

    # Fourth L-moment

    comb5 = np.zeros(n)
    for i in range(n):
        comb5[i] = comb(i, 3)
    coefl4 = 0.25 / comb(n, 4)
    sum_xtrans[:] = 0
    for i in nb.prange(n):
        sum_xtrans += (
            comb5[i]
            - 3 * comb3[i] * comb1[n - 1 - i]
            + 3 * comb1[i] * comb3[n - 1 - i]
            - comb5[n - 1 - i]
        ) * x[i]
    l4 = coefl4 * sum_xtrans / l2

    if nmom == 4:
        return np.stack((l1, l2, l3, l4))

    # Fifth L-moment

    comb7 = np.zeros(n)
    for i in range(n):
        comb7[i] = comb(i, 4)
    coefl5 = 0.2 / comb(n, 5)
    sum_xtrans[:] = 0
    for i in nb.prange(n):
        sum_xtrans += (
            comb7[i]
            - 4 * comb5[i] * comb1[n - 1 - i]
            + 6 * comb3[i] * comb3[n - 1 - i]
            - 4 * comb1[i] * comb5[n - 1 - i]
            + comb7[n - 1 - i]
        ) * x[i]

    l5 = coefl5 * sum_xtrans / l2

    return np.stack((l1, l2, l3, l4, l5))
