REGISTRY = {}

from .deconvolution_reduced import BlindRL
REGISTRY['BlindRL'] = BlindRL

from .care import CAREDeconv
REGISTRY['csbdeep'] = CAREDeconv

from .mu_net import Mu_Net
REGISTRY['mu-net'] = Mu_Net

from .wb_backprojector import WBBackProjectorDeconv
REGISTRY['WB'] = WBBackProjectorDeconv