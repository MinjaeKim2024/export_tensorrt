# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

from __future__ import absolute_import

from boxmot.appearance.backbones.osnet_ain import (osnet_ain_x1_0)

NR_CLASSES_DICT = {'market1501': 751, 'duke': 702, 'veri': 576, 'vehicleid': 576}


__model_factory = {"osnet_ain_x1_0": osnet_ain_x1_0}


def get_nr_classes(weigths):
    num_classes = [value for key, value in NR_CLASSES_DICT.items() if key in str(weigths.name)]
    if len(num_classes) == 0:
        num_classes = 1
    else:
        num_classes = num_classes[0]
    return num_classes


def build_model(name, num_classes, loss="softmax", pretrained=True, use_gpu=True):
    """A function wrapper for building a model.

    Args:
        name (str): model name.
        num_classes (int): number of training identities.
        loss (str, optional): loss function to optimize the model. Currently
            supports "softmax" and "triplet". Default is "softmax".
        pretrained (bool, optional): whether to load ImageNet-pretrained weights.
            Default is True.
        use_gpu (bool, optional): whether to use gpu. Default is True.

    Returns:
        nn.Module

    Examples::
        >>> from torchreid import models
        >>> model = models.build_model('resnet50', 751, loss='softmax')
    """
    avai_models = list(__model_factory.keys())
    if name not in avai_models:
        raise KeyError("Unknown model: {}. Must be one of {}".format(name, avai_models))
    if 'clip' in name:
        from boxmot.appearance.backbones.clip.config.defaults import _C as cfg
        return __model_factory[name](cfg, num_class=num_classes, camera_num=2, view_num=1)
    
    return __model_factory[name](
        num_classes=num_classes, loss=loss, pretrained=pretrained, use_gpu=use_gpu
    )
