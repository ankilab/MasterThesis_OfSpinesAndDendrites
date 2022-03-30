# Source: https://github.com/CSBDeep/CSBDeep/blob/fc4479f74d04eebcf871e2957e0ed9dff65a03a4/csbdeep/data/prepare.py
import numpy as np


class Normalizer:
    """
    Super-class for implemented normalization-approaches. Every derived class overwrited the normalize method according
    to its functionality.

    """

    def __init__(self):
        pass

    def normalize(self, **kwargs):
        """
        Implemented in derived classes

        """

        return NotImplementedError


class PercentileNormalizer(Normalizer):
    """
    Percentile-based image normalization. Adjusted from CSBDeep package.

    """

    def __init__(self, pmin=2, pmax=99.9, dtype=np.float32, **kwargs):
        """
        Initialize object for percentile-based image normalization

        :param pmin: Low percentile, defaults to 2
        :type pmin: float, optional
        :param pmax: High percentile, defaults to 99.9
        :type pmax: float, optional
        :param dtype: Data type after normalization, defaults to np.float32
        :type dtype: type, optional
        :param kwargs: Keyword arguments for :func:`csbdeep.utils.normalize_mi_ma`.
        :type kwargs: dict, optional
        """
        super().__init__()
        assert (np.isscalar(pmin) and np.isscalar(pmax) and 0 <= pmin < pmax <= 100)
        self.pmin = pmin
        self.pmax = pmax
        self.dtype = dtype
        self.kwargs = kwargs

    def normalize(self, x):
        """
        Percentile-based normalization of raw input image.

        :param x: Input to be normalized
        :type x: nd.array
        """
        self.mi = np.percentile(x,self.pmin,keepdims=True).astype(self.dtype,copy=False)
        self.ma = np.percentile(x,self.pmax,keepdims=True).astype(self.dtype,copy=False)
        return self.normalize_mi_ma(x, self.mi, self.ma, dtype=self.dtype, **self.kwargs)

    def normalize_mi_ma(self, x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
        """
        Normalize input given minimum and maximum values.

        :param x: Input to be normalized
        :type x: nd.array
        :param mi: Minimum value
        :type mi: float or ndarray
        :param ma: Maximum value
        :type ma: float or ndarray
        :param clip: Flag whether tio clip the output to range of 0 to 1, defaults to False
        :type clip: bool, optional
        :param eps: Constant to avoid division by zero, defaults to 1e-20
        :type eps: float, optional
        :param dtype: Data type after normalization, defaults to np.float32
        :type dtype: type, optional
        :return: Normalized input
        :rtype: nd.array
        """

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
        """
        Affine rescaling of x, such that the mean squared error to target is minimal.

        :param x: Moving image
        :type x: nd.array
        :param target: Target image
        :type target: nd.array
        """
        cov = np.cov(x.flatten(), target.flatten())
        alpha = cov[0, 1] / (cov[0, 0] + 1e-10)
        beta = target.mean() - alpha * x.mean()
        return alpha * x + beta


class MinMaxNormalizer(Normalizer):
    """
    Min-Max-Normalization.

    """

    def __init__(self):
        super().__init__()

    def normalize(self, img, min=None, max=None):
        """
        Normalize input to range of 0 and 1.

        :param img: Input image
        :param min: Minimum value
        :type min: float, optional
        :param max: Maximum value
        :type max: float, optional
        :return: Normalized Input
        :rtype: nd.array
        """
        if min is None or min >img.min():
            min = img.min()
        img = img-min
        if max is None or max <img.max():
            max= img.max()

        return img/max


class Rescaler(Normalizer):
    """
    Rescale input to new value range.

    """

    def __init__(self):
        super().__init__()

    def normalize(self, img, new_min=0, new_max=1):
        """
        Rescale input to new value range.

        :param img: Input image
        :param new_min: Minimum value after rescaljng
        :type new_min: float, optional
        :param new_max: Maximum value  after rescaljng
        :type new_max: float, optional
        :return: Normalized Input
        :rtype: nd.array
        """
        img -= np.min(img)
        img /= np.max(img)
        img /= np.max(img)
        img *= (new_max-new_min)
        img += new_min
        return img



