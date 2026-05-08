from .trajworld_dataset import TrajWorldDataset
from .tdm_dataset import TDMDataset
from .mlp_ensemble_dataset import MLPEnsembleDataset
from .westworld_dataset import WestWorldDataset
__all__ = {
    'WestWorld': WestWorldDataset,
    'Trajworld': TrajWorldDataset,
    "TDM": TDMDataset,
    "MLPEnsemble": MLPEnsembleDataset,
    # more model in the future
}


def build_dataset(config, val=False):
    dataset_class = __all__[config.method.model_name]
    dataset = dataset_class(config=config, is_validation=val)
    return dataset
