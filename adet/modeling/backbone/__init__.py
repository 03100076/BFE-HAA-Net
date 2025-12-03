from .fpn import build_fcos_resnet_fpn_backbone   #FPN
from .vovnet import build_vovnet_fpn_backbone, build_vovnet_backbone
from .dla import build_fcos_dla_fpn_backbone
from .resnet_lpf import build_resnet_lpf_backbone
#from .bifpn import build_fcos_resnet_bifpn_backbone  #BiFPN
#from .our_v1 import build_fcos_resnet_bifpn_backbone_v1  #EBUP
#from .our_v2 import build_fcos_resnet_bifpn_backbone_v2  #PANet
#from .our_v4 import build_fcos_resnet_fpn_backbone_v4   #ATDP
#from .nofpn import build_fcos_resnet_nofpn_backbone    #baseline
#from .our_v3_2_new import build_fcos_resnet_bifpn_backbone_v3_2 #MCAM
#from .our_v3_1_new import build_fcos_resnet_bifpn_backbone_v3_1  #MSAM
#from .our_v3_new import build_fcos_resnet_bifpn_backbone_v3 #BFE-HAA-Net
