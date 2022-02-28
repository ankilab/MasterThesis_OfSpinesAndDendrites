# Source: https://github.com/CSBDeep/CSBDeep/blob/fc4479f74d04eebcf871e2957e0ed9dff65a03a4/csbdeep/data/prepare.py
import numpy as np

class Normalizer:

    def __init__(self):
        pass

    def normalize(self, **kwargs):
        return NotImplementedError

class PercentileNormalizer(Normalizer):
    """Percentile-based image normalization.
    Parameters
    ----------
    pmin : float
        Low percentile.
    pmax : float
        High percentile.
    do_after : bool
        Flag to indicate whether to undo normalization (original data type will not be restored).
    dtype : type
        Data type after normalization.
    kwargs : dict
        Keyword arguments for :func:`csbdeep.utils.normalize_mi_ma`.
    """

    def __init__(self, pmin=2, pmax=99.9, do_after=True, dtype=np.float32, **kwargs):
        """TODO."""
        super().__init__()
        assert (np.isscalar(pmin) and np.isscalar(pmax) and 0 <= pmin < pmax <= 100)
        self.pmin = pmin
        self.pmax = pmax
        self._do_after = do_after
        self.dtype = dtype
        self.kwargs = kwargs

    def normalize(self, x):
        """Percentile-based normalization of raw input image.
        See :func:`csbdeep.predict.Normalizer.before` for parameter descriptions.
        Note that percentiles are computed individually for each channel (if present in `axes`).
        """
        self.mi = np.percentile(x,self.pmin,keepdims=True).astype(self.dtype,copy=False)
        self.ma = np.percentile(x,self.pmax,keepdims=True).astype(self.dtype,copy=False)
        return self.normalize_mi_ma(x, self.mi, self.ma, dtype=self.dtype, **self.kwargs)

    def normalize_mi_ma(self, x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
        if dtype is not None:
            x = x.astype(dtype, copy=False)
            mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
            ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy=False)
            eps = dtype(eps)

        try:
            import numexpr
            x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")
        except ImportError:
            x = (x - mi) / (ma - mi + eps)

        if clip:
            x = np.clip(x, 0, 1)

        return x

    def normalize_minmse(self, x, target):
        """Affine rescaling of x, such that the mean squared error to target is minimal."""
        cov = np.cov(x.flatten(), target.flatten())
        alpha = cov[0, 1] / (cov[0, 0] + 1e-10)
        beta = target.mean() - alpha * x.mean()
        return alpha * x + beta


class MinMaxNormalizer(Normalizer):
    def __init__(self):
        super().__init__()

    def normalize(self, img, min=None, max=None):
        if min is None:
            min = img.min()
        img = img-min
        if max is None:
            max= img.max()

        return img/max


class Rescaler(Normalizer):
    def __init__(self):
        super().__init__()

    def normalize(self, img, new_min=0, new_max=1):
        img -= np.min(img)
        img /= np.max(img)
        img /= np.max(img)
        img *= (new_max-new_min)
        img += new_min
        return img



