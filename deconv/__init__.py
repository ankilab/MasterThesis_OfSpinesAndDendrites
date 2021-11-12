REGISTRY = {}

from .blind_rl import BlindRL
REGISTRY['BlindRL'] = BlindRL

from .care import CAREDeconv
REGISTRY['csbdeep'] = CAREDeconv

from .mu_net import Mu_Net
REGISTRY['mu-net'] = Mu_Net