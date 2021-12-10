# https://stackoverflow.com/questions/47623126/convolution-of-3d-numpy-arrays
import numpy as np
cimport numpy as np
cimport cython

np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.embedsignature(True)
def myconvolve(np.ndarray[np.float64_t, ndim=3] A,
               np.ndarray[np.float64_t, ndim=3] B):
    cdef:
        int n, m, i, j
        int NAx = A.shape[0], NAy = A.shape[1], NAz = A.shape[2]
        int NBx = A.shape[0], NBy = A.shape[1], NBz = A.shape[2]
        int Deg = NAz + NBz - 1;
        np.ndarray[np.float64_t, ndim=3] C = np.zeros((NAx, NBy, Deg));
    assert((NAx == NBx) and (NAy == NBy))

    for n in range(0, (Deg)):
        for m in range(0, n + 1):
            if ((n - m) < NAz and m < NBz):
                for i in range(0, NAx):
                    for j in range(0, NAy):
                        C[i, j, n] = C[i, j, n] + A[i, j, (n - m)] * B[i, j, m]

    return C