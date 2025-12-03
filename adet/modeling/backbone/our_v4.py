from torch import nn
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init
import math
from detectron2.modeling.backbone import FPN, build_resnet_backbone
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY

from .resnet_lpf import build_resnet_lpf_backbone
from .resnet_interval import build_resnet_interval_backbone
from .mobilenet import build_mnv2_backbone

class FPN_v4(FPN):
    """
    This module implements :paper:`FPN`.
    It creates pyramid features built on top of some input feature maps.
    """
    def __init__(
        self,
        bottom_up,
        in_features,
        out_channels,
        norm="",
        top_block=None,
        fuse_type="sum",
        square_pad=0,
    ):
        super(FPN_v4, self).__init__(
            bottom_up=bottom_up,
            in_features=in_features,
            out_channels=out_channels,
            norm=norm,
            top_block=top_block,
            fuse_type=fuse_type,
            square_pad=square_pad)

        # Feature map strides and channels from the bottom up network (e.g. ResNet)
        input_shapes = bottom_up.output_shape()
        strides = [input_shapes[f].stride for f in in_features] #[8, 16, 32]
        in_channels_per_feature = [input_shapes[f].channels for f in in_features] #[512, 1024, 2048]

        cross_convs = []
        use_bias = norm == ""
        for idx, in_channels in enumerate(in_channels_per_feature):
            if idx == 0 or idx == len(in_channels_per_feature) - 1:
                cross_convs.append(None)
                continue
            cross_norm = get_norm(norm, out_channels)

            cross_conv = Conv2d(
                in_channels, out_channels, kernel_size=1, bias=use_bias, norm=cross_norm
            )
            weight_init.c2_xavier_fill(cross_conv)
            
            stage = int(math.log2(strides[idx]))
            self.add_module("fpn_cross{}".format(stage), cross_conv)

            cross_convs.append(cross_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.cross_convs = cross_convs[::-1]

    def forward(self, x):
        """
        Args:
            input (dict[str->Tensor]): mapping feature map name (e.g., "res5") to
                feature map tensor for each feature level in high to low resolution order.

        Returns:
            dict[str->Tensor]:
                mapping from feature map name to FPN feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        bottom_up_features = self.bottom_up(x)
        results = []
        prev_features = self.lateral_convs[0](bottom_up_features[self.in_features[-1]])
        cross_features = None
        results.append(self.output_convs[0](prev_features))

        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, (lateral_conv, output_conv, cross_conv) in enumerate(
            zip(self.lateral_convs, self.output_convs, self.cross_convs)
        ):
            # Slicing of ModuleList is not supported https://github.com/pytorch/pytorch/issues/47336
            # Therefore we loop over all modules but skip the first one
            if idx > 0:
                features = self.in_features[-idx - 1]
                features = bottom_up_features[features]
                top_down_features = F.interpolate(prev_features, scale_factor=2.0, mode="nearest")
                lateral_features = lateral_conv(features)
                if cross_features is not None:
                    prev_features = lateral_features + top_down_features + cross_features
                else:
                    prev_features = lateral_features + top_down_features
                if self._fuse_type == "avg":
                    prev_features /= (2 + (cross_features is not None))
                results.insert(0, output_conv(prev_features))
                
                if cross_conv is not None:
                    cross_features = cross_conv(features)
                    cross_features = F.interpolate(cross_features, scale_factor=2.0, mode="nearest")
                else:
                    cross_features = None

        if self.top_block is not None:
            if self.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[self.top_block.in_feature]
            else:
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))
        assert len(self._out_features) == len(results)
        return {f: res for f, res in zip(self._out_features, results)}

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


@BACKBONE_REGISTRY.register()
def build_fcos_resnet_fpn_backbone_v4(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    if cfg.MODEL.BACKBONE.ANTI_ALIAS:
        bottom_up = build_resnet_lpf_backbone(cfg, input_shape)
    elif cfg.MODEL.RESNETS.DEFORM_INTERVAL > 1:
        bottom_up = build_resnet_interval_backbone(cfg, input_shape)
    elif cfg.MODEL.MOBILENET:
        bottom_up = build_mnv2_backbone(cfg, input_shape)
    else:
        bottom_up = build_resnet_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES # ['res2', 'res3', 'res4', 'res5']
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    top_levels = cfg.MODEL.FCOS.TOP_LEVELS # 设置top_levels=0
    in_channels_top = out_channels
    if top_levels == 2:
        top_block = LastLevelP6P7(in_channels_top, out_channels, "p5")
    if top_levels == 1:
        top_block = LastLevelP6(in_channels_top, out_channels, "p5")
    elif top_levels == 0:
        top_block = None
    backbone = FPN_v4(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=top_block,
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone
