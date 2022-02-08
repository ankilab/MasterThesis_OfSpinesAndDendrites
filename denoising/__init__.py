from .denoiser import GaussianFilter, BilateralFilter, Neighbor2Neighbor, NLMeans, BM3D

REGISTRY = {}

REGISTRY['Gauss'] = GaussianFilter
REGISTRY['Bilateral'] = BilateralFilter
REGISTRY['n2n'] =Neighbor2Neighbor
REGISTRY['nlm'] =NLMeans
REGISTRY['bm3d'] =BM3D