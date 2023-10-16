from torch import nn, Tensor
from typing import Callable, Dict, List, Optional, Union
from torchvision.ops.feature_pyramid_network import ExtraFPNBlock, FeaturePyramidNetwork, LastLevelMaxPool
from torchvision.models._utils import handle_legacy_interface, IntermediateLayerGetter
from options.opts import get_training_arguments
from options.utils import load_config_file
from cvnets.models.classification import arguments_classification, build_classification_model
from torchvision.ops.feature_pyramid_network import LastLevelP6P7
from torchvision.models.detection.anchor_utils import AnchorGenerator
#from torchvision.models.detection import RetinaNet
from vision_models.retinanet import RetinaNet

class BackboneWithFPN(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        return_layers: Dict[str, str],
        in_channels_list: List[int],
        out_channels: int,
        extra_blocks: Optional[ExtraFPNBlock] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()

        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()

        self.body = IntermediateLayerGetter(
            backbone, return_layers=return_layers)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
            norm_layer=norm_layer,
        )
        self.out_channels = out_channels

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.body(x)
        x = self.fpn(x)
        return x
    

def _mobilevit_fpn_extractor(
    backbone,
    trainable_layers: int,
    returned_layers: Optional[List[int]] = None,
    extra_blocks: Optional[ExtraFPNBlock] = None,
    norm_layer: Optional[Callable[..., nn.Module]] = None,
):
    layers_list = ["conv_1x1_exp", "layer_5", "layer_4", "layer_3", "layer_2", "layer_1", "conv_1"]
    layers_to_train = layers_list[:trainable_layers]
    for name, parameter in backbone.named_modules():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)
    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()
    if returned_layers is None:
        returned_layers = [1,2,3,4,5]
    if min(returned_layers)<=0 or max(returned_layers)>=6:
        raise ValueError(f"Each returned layer should be in the range [1,5]. Got {returned_layers}")
    return_layers = {f"layer_{k}": str(v) for v,k in enumerate(returned_layers)}
    in_channels_list = []
    for layer in layers_list[::-1]:
        if layer.split('_')[0] == 'layer':
            in_channels_list.append(getattr(backbone, layer)[0].out_channels)
    return_in_channels_list = []
    for idx,channel  in enumerate(in_channels_list):
        idx = idx + 1
        if ( idx in returned_layers): return_in_channels_list.append(channel)
    out_channels=256
    return BackboneWithFPN(backbone=backbone, return_layers=return_layers, in_channels_list=return_in_channels_list, out_channels=out_channels, extra_blocks=extra_blocks, norm_layer=norm_layer)

def _default_anchorgen():
    #anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [8, 16, 32, 64, 128, 256])
    #[4,8,16,32,64]
    anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [32, 64, 128, 256, 512])
    aspect_ratios = ((1/1, 2/1, 1/2,),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
    return anchor_generator

def model_generator(): 
    opts = get_training_arguments()
    setattr(opts, "common.config_file", "config/detection/ssd_mobilevitv3_xx_small_320.yaml")
    opts = load_config_file(opts=opts)

    setattr(opts, "model.detection.n_classes", 81)
    setattr(opts, "dataset.workers", 0)
    for k,v in vars(opts).items():
        print(k, " : ", v)
    
    mobilevit = build_classification_model(opts=opts)

    backbone_fpn = _mobilevit_fpn_extractor(backbone=mobilevit, trainable_layers=0, returned_layers=[3,4,5], extra_blocks=LastLevelP6P7(256, 256))
    
    anchor_generator = _default_anchorgen()
    model_main = RetinaNet(backbone=backbone_fpn, num_classes=2, anchor_generator=anchor_generator, nms_thresh=0.95, detections_per_img=999999, score_thresh=0.1)
    
    return model_main, backbone_fpn, mobilevit