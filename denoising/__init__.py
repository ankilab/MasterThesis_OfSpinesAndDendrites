from .denoiser import GaussianFilter, BilateralFilter
from .denoiser import Neighbor2Neighbor
# from .denoiser import Neighbor2Neighbor, NLMeans, BM3D

REGISTRY = {}

REGISTRY['Gauss'] = GaussianFilter
REGISTRY['Bilateral'] = BilateralFilter
REGISTRY['n2n'] =Neighbor2Neighbor
# REGISTRY['nlm'] =NLMeans
# REGISTRY['bm3d'] =BM3D