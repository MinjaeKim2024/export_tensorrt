# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license


from collections import OrderedDict

import torch

from boxmot.utils import logger as LOGGER

__model_types = [
    "resnet50",
    "resnet101",
    "mlfn",
    "hacnn",
    "mobilenetv2_x1_0",
    "mobilenetv2_x1_4",
    "osnet_x1_0",
    "osnet_x0_75",
    "osnet_x0_5",
    "osnet_x0_25",
    "osnet_ibn_x1_0",
    "osnet_ain_x1_0",
    "lmbn_n",
    "clip",
]

lmbn_loc = 'https://github.com/mikel-brostrom/yolov8_tracking/releases/download/v9.0/'



def get_model_name(model):
    for x in __model_types:
        if x in model.name:
            return x
    return None

def load_pretrained_weights(model, weight_path):
    r"""Loads pretrianed weights to model.

    Features::
        - Incompatible layers (unmatched in name or size) will be ignored.
        - Can automatically deal with keys containing "module.".

    Args:
        model (nn.Module): network model.
        weight_path (str): path to pretrained weights.

    Examples::
        >>> from boxmot.appearance.backbones import build_model
        >>> from boxmot.appearance.reid_model_factory import load_pretrained_weights
        >>> weight_path = 'log/my_model/model-best.pth.tar'
        >>> model = build_model()
        >>> load_pretrained_weights(model, weight_path)
    """
    if not torch.cuda.is_available():
        checkpoint = torch.load(weight_path, map_location=torch.device("cpu"))
    else:
        checkpoint = torch.load(weight_path)

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model_dict = model.state_dict()

    if "lmbn" in str(weight_path):
        model.load_state_dict(model_dict, strict=True)
    elif "clip" in str(weight_path):
        def forward_override(self, x: torch.Tensor, cv_emb=None, old_forward=None):
            _, image_features, image_features_proj = old_forward(x, cv_emb)
            return torch.cat([image_features[:, 0], image_features_proj[:, 0]], dim=1)
        # print('model.load_param(str(weight_path))', str(weight_path))
        model.load_param(str(weight_path))
        model = model.image_encoder
        # old_forward = model.forward
        # model.forward = lambda *args, **kwargs: forward_override(model, old_forward=old_forward, *args, **kwargs)
        LOGGER.success(
            f'Successfully loaded pretrained weights from "{weight_path}"'
        )
    else:
        new_state_dict = OrderedDict()
        matched_layers, discarded_layers = [], []

        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[7:]  # discard module.

            if k in model_dict and model_dict[k].size() == v.size():
                new_state_dict[k] = v
                matched_layers.append(k)
            else:
                discarded_layers.append(k)

        model_dict.update(new_state_dict)
        model.load_state_dict(model_dict)

        if len(matched_layers) == 0:
            LOGGER.warning(
                f'The pretrained weights "{weight_path}" cannot be loaded, '
                "please check the key names manually "
                "(** ignored and continue **)"
            )
        else:
            LOGGER.success(
                f'Successfully loaded pretrained weights from "{weight_path}"'
            )
            if len(discarded_layers) > 0:
                LOGGER.warning(
                    "The following layers are discarded "
                    f"due to unmatched keys or layer size: {*discarded_layers,}"
                )
