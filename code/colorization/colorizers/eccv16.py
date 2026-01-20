
import torch
import torch.nn as nn
import numpy as np
from IPython import embed

from .base_color import *
from .ila_block import ILA

class ECCVGenerator(BaseColor):
    def __init__(self, norm_layer=nn.BatchNorm2d, use_ila=False, ila_reduction=4, ila_use_dw_conv=True):
        """
        ECCV16 Colorization Generator with optional ILA (Inter-Leaved Light Attention) blocks.
        
        Args:
            norm_layer: Normalization layer to use (default: nn.BatchNorm2d)
            use_ila (bool): If True, interleave ILA blocks between encoder stages.
                When False, behavior is identical to baseline ECCV16. Default: False
            ila_reduction (int): Channel reduction factor for ILA blocks. Default: 4
            ila_use_dw_conv (bool): Whether ILA blocks use depthwise convolution. Default: True
        """
        super(ECCVGenerator, self).__init__()
        
        self.use_ila = use_ila

        model1=[nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True),]
        model1+=[nn.ReLU(True),]
        model1+=[nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),]
        model1+=[nn.ReLU(True),]
        model1+=[norm_layer(64),]

        model2=[nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),]
        model2+=[nn.ReLU(True),]
        model2+=[nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),]
        model2+=[nn.ReLU(True),]
        model2+=[norm_layer(128),]

        model3=[nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model3+=[nn.ReLU(True),]
        model3+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model3+=[nn.ReLU(True),]
        model3+=[nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),]
        model3+=[nn.ReLU(True),]
        model3+=[norm_layer(256),]

        model4=[nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model4+=[nn.ReLU(True),]
        model4+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model4+=[nn.ReLU(True),]
        model4+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model4+=[nn.ReLU(True),]
        model4+=[norm_layer(512),]

        model5=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model5+=[nn.ReLU(True),]
        model5+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model5+=[nn.ReLU(True),]
        model5+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model5+=[nn.ReLU(True),]
        model5+=[norm_layer(512),]

        model6=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model6+=[nn.ReLU(True),]
        model6+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model6+=[nn.ReLU(True),]
        model6+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model6+=[nn.ReLU(True),]
        model6+=[norm_layer(512),]

        model7=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model7+=[nn.ReLU(True),]
        model7+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model7+=[nn.ReLU(True),]
        model7+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model7+=[nn.ReLU(True),]
        model7+=[norm_layer(512),]

        model8=[nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True),]
        model8+=[nn.ReLU(True),]
        model8+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model8+=[nn.ReLU(True),]
        model8+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model8+=[nn.ReLU(True),]

        model8+=[nn.Conv2d(256, 313, kernel_size=1, stride=1, padding=0, bias=True),]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)
        self.model8 = nn.Sequential(*model8)
        
        # Inter-leave ILA blocks between encoder stages when use_ila=True
        # Insert after model2 (64->128), model3 (128->256), model4 (256->512)
        # This allows global context injection at multiple scales
        if use_ila:
            self.ila1 = ILA(in_channels=128, reduction=ila_reduction, use_dw_conv=ila_use_dw_conv)
            self.ila2 = ILA(in_channels=256, reduction=ila_reduction, use_dw_conv=ila_use_dw_conv)
            self.ila3 = ILA(in_channels=512, reduction=ila_reduction, use_dw_conv=ila_use_dw_conv)
        else:
            # Identity modules to maintain compatibility
            self.ila1 = nn.Identity()
            self.ila2 = nn.Identity()
            self.ila3 = nn.Identity()

        self.softmax = nn.Softmax(dim=1)
        self.model_out = nn.Conv2d(313, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=False)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, input_l):
        conv1_2 = self.model1(self.normalize_l(input_l))
        conv2_2 = self.model2(conv1_2)
        
        # Inter-leave ILA: ConvBlock -> ILA -> ConvBlock -> ILA
        if self.use_ila:
            conv2_2 = self.ila1(conv2_2)  # ILA after model2 (128 channels)
        
        conv3_3 = self.model3(conv2_2)
        
        if self.use_ila:
            conv3_3 = self.ila2(conv3_3)  # ILA after model3 (256 channels)
        
        conv4_3 = self.model4(conv3_3)
        
        if self.use_ila:
            conv4_3 = self.ila3(conv4_3)  # ILA after model4 (512 channels)
        
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)
        conv8_3 = self.model8(conv7_3)
        out_reg = self.model_out(self.softmax(conv8_3))

        return self.unnormalize_ab(self.upsample4(out_reg))

def eccv16(pretrained=True, use_ila=False, ila_reduction=4, ila_use_dw_conv=True):
	"""
	Create ECCV16 colorization model.
	
	Args:
		pretrained (bool): If True, load pretrained weights. Default: True
		use_ila (bool): If True, use ILA blocks. Note: pretrained weights are not
			compatible with ILA. Default: False
		ila_reduction (int): Channel reduction for ILA blocks. Default: 4
		ila_use_dw_conv (bool): Whether ILA uses depthwise conv. Default: True
	
	Returns:
		ECCVGenerator: ECCV16 model instance
	"""
	model = ECCVGenerator(use_ila=use_ila, ila_reduction=ila_reduction, ila_use_dw_conv=ila_use_dw_conv)
	if(pretrained):
		if use_ila:
			import warnings
			warnings.warn(
				"Pretrained weights are not compatible with ILA. "
				"Model will be initialized with random weights. "
				"Set use_ila=False to use pretrained weights.",
				UserWarning
			)
		else:
			import torch.utils.model_zoo as model_zoo
			model.load_state_dict(model_zoo.load_url('https://colorizers.s3.us-east-2.amazonaws.com/colorization_release_v2-9b330a0b.pth',map_location='cpu',check_hash=True))
	return model
