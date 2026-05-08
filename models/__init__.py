from .Trajworld.trajworld import TrajWorld
from .WestWorld.westworld import WestWorld
from .TDM.trm_tdm_torch import TDM
from .MLPEnsemble.mlp_ensemble_torch import MLPEnsemble
__all__ = {
    'WestWorld': WestWorld,
    'Trajworld': TrajWorld,
    "TDM" : TDM,
    "MLPEnsemble" : MLPEnsemble
}


def build_model(config):
    model = __all__[config.method.model_name](
        config=config
    )

    return model
