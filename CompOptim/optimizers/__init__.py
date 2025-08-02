from .DAdaGrad import *
from .DADAM import *
from .DAMSCo import *
from .DaSHCo import *
from .CDProxSGD import *

# map from string → class
_opt_map = {
    "DAdaGrad":    DAdaGrad,
    "DADAM":       DADAM,
    "DAMSCo":      DAMSCo,
    "DaSHCo":      DaSHCo,
    "CDProxSGD":   CDProxSGD,
}

def get_optimizer(name, *args, **kwargs):
    """
    Given the class-name string, return an instance of that optimizer.
    """
    try:
        cls = _opt_map[name]
    except KeyError:
        raise ValueError(f"Unknown optimizer '{name}'.  "
                         f"Available: {list(_opt_map)}")
    return cls(*args, **kwargs)