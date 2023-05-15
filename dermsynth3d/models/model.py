from collections import OrderedDict, namedtuple

import torch
import torch.nn as nn
import torchvision.transforms.functional as F

from torchvision.models.vgg import vgg16
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from dermsynth3d.utils.anatomy import SimpleAnatomy


class SkinDeepLabV3(nn.Module):
    def __init__(
        self, multi_head: bool, freeze_backbone: bool, include_anatomy: bool = True
    ):
        super(SkinDeepLabV3, self).__init__()

        self.model = deeplabv3_resnet50(pretrained=True)
        self.model.classifier = None
        self.model.aux_classifier = None
        self.model.segmenter = None
        self.model.anatomy = None
        self.model.depth = None
        self.multi_head = multi_head
        self.freeze_backbone = freeze_backbone
        self.include_anatomy = include_anatomy
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        # Three channels for skin, lesion, background segmentation.
        self.n_seg_channels = 3

        # The number of anatomy channels.
        self.n_anatomy_channels = 0
        if self.include_anatomy:
            self.n_anatomy_channels = SimpleAnatomy.n_labels

        # One channel for depth.
        self.n_depth_channels = 0
        # Total number of channels.
        self.n_channels = (
            self.n_seg_channels + self.n_anatomy_channels + self.n_depth_channels
        )

        if self.multi_head:
            self.model.segmenter = DeepLabHead(2048, self.n_seg_channels)
            self.model.anatomy = DeepLabHead(2048, self.n_anatomy_channels)
            self.model.depth = DeepLabHead(2048, self.n_depth_channels)
        else:
            self.model.classifier = DeepLabHead(2048, self.n_channels)

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.model.backbone(x)
        result = OrderedDict()
        f = features["out"]
        # result["f"] = f

        if self.multi_head is False:
            x = self.model.classifier(f)
            x = torch.nn.functional.interpolate(
                x, size=input_shape, mode="bilinear", align_corners=False
            )
            result["segmenter"] = x[:, : self.n_seg_channels, :, :]
            if self.include_anatomy:
                result["anatomy"] = x[
                    :,
                    self.n_seg_channels : (
                        self.n_seg_channels + self.n_anatomy_channels
                    ),
                    :,
                    :,
                ]
            # result["depth"] = x[:, -1, :, :]

        if self.model.segmenter is not None:
            x = self.model.segmenter(f)
            x = torch.nn.functional.interpolate(
                x, size=input_shape, mode="bilinear", align_corners=False
            )
            result["segmenter"] = x

        if self.model.anatomy is not None:
            x = self.model.anatomy(f)
            x = torch.nn.functional.interpolate(
                x, size=input_shape, mode="bilinear", align_corners=False
            )
            result["anatomy"] = x

        if self.model.depth is not None:
            x = self.model.depth(f)
            x = torch.nn.functional.interpolate(
                x, size=input_shape, mode="bilinear", align_corners=False
            )
            result["depth"] = x

        return result


class Vgg16Seg(nn.Module):
    def __init__(self, n_classes: int, img_size: tuple):
        super(Vgg16Seg, self).__init__()
        self.features = vgg16(pretrained=True).features
        kernel_size = (1, 1)
        self.img_size = img_size

        self.conv2d_layer3 = nn.Conv2d(256, 64, kernel_size)
        self.conv2d_layer4 = nn.Conv2d(
            512,
            64,
            kernel_size,
        )
        self.conv2d_layer5 = nn.Conv2d(
            512,
            64,
            kernel_size,
        )
        self.conv2d_layer6 = nn.Conv2d(
            512,
            64,
            kernel_size,
        )

        self.conv2d_output = nn.Conv2d(
            512 + 64 + 64,
            n_classes,
            kernel_size,
        )

        self.conv2d_spatial7 = nn.Conv2d(
            64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )

        self.conv2d_spatial8 = nn.Conv2d(
            64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )

        self.maxpool = nn.MaxPool2d((2, 2))
        self.relu = nn.ReLU()  # nn.LeakyReLU(negative_slope=0.1)
        self.sigmoid = nn.Sigmoid()
        self.softmax2d = torch.nn.Softmax2d()

    def forward(self, x):
        input_shape = x.shape[-2:]

        for layer_id, layer in enumerate(self.features):
            x = layer(x)
            if layer_id == 0:
                layer0_responses = x
            if layer_id == 2:
                layer1_responses = x
            elif layer_id == 7:
                layer2_responses = x
            elif layer_id == 14:
                layer3_responses = x
            elif layer_id == 21:
                layer4_responses = x
            elif layer_id == 28:
                layer5_responses = x
            elif layer_id == 30:
                layer6_responses = x

        layer0_responses = self.relu(layer0_responses)

        layer1_responses = self.relu(layer1_responses)

        layer2_responses = F.resize(layer2_responses, size=input_shape)
        layer2_responses = self.relu(layer2_responses)

        conv_response_3 = self.conv2d_layer3(layer3_responses)
        conv_response_3 = F.resize(conv_response_3, size=input_shape)
        conv_response_3 = self.relu(conv_response_3)

        conv_response_4 = self.conv2d_layer4(layer4_responses)
        conv_response_4 = F.resize(conv_response_4, size=input_shape)
        conv_response_4 = self.relu(conv_response_4)

        conv_response_5 = self.conv2d_layer5(layer5_responses)
        conv_response_5 = F.resize(conv_response_5, size=input_shape)
        conv_response_5 = self.relu(conv_response_5)

        conv_response_6_orig = self.conv2d_layer6(layer6_responses)
        conv_response_6 = F.resize(conv_response_6_orig, size=input_shape)
        conv_response_6 = self.relu(conv_response_6)
        conv_response_6_act = self.relu(conv_response_6_orig)
        conv_response_6_max = self.maxpool(conv_response_6_act)

        conv_response_7_orig = self.conv2d_spatial7(conv_response_6_max)
        conv_response_7 = F.resize(conv_response_7_orig, size=input_shape)
        conv_response_7 = self.relu(conv_response_7)
        conv_response_7_act = self.relu(conv_response_7_orig)
        conv_response_7_max = self.maxpool(conv_response_7_act)

        conv_response_8_orig = self.conv2d_spatial8(conv_response_7_max)
        conv_response_8 = F.resize(conv_response_8_orig, size=input_shape)
        conv_response_8 = self.relu(conv_response_8)

        x = torch.cat(
            (
                layer0_responses,
                layer1_responses,
                layer2_responses,
                conv_response_3,
                conv_response_4,
                conv_response_5,
                conv_response_6,
                conv_response_7,
                conv_response_8,
            ),
            1,
        )
        x = self.conv2d_output(x)

        result = OrderedDict()
        result["out"] = x

        return result


class MeanShift(torch.nn.Conv2d):
    def __init__(self, gpu_id):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        rgb_range = 1
        rgb_mean = (0.4488, 0.4371, 0.4040)
        # rgb_std=(1.0, 1.0, 1.0)
        rgb_std = (0.2, 0.2, 0.2)
        sign = -1
        std = torch.Tensor(rgb_std).to(gpu_id)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1).to(gpu_id) / std.view(
            3, 1, 1, 1
        )
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean).to(gpu_id) / std
        for p in self.parameters():
            p.requires_grad = False


class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple(
            "VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3"]
        )
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out


def faster_rcnn_texture_model(
    device,
    num_classes=2,
    max_input_size=4096,
    anchor_generator=None,
    pretrained_backbone=False,
):
    if anchor_generator is None:
        sizes = ((32,), (24,), (22,), (16,), (8,))
        anchor_generator = AnchorGenerator(
            sizes=sizes,
            aspect_ratios=((0.75, 1, 1.25),) * len(sizes),
        )

    model = fasterrcnn_resnet50_fpn(
        pretrained_backbone=pretrained_backbone, max_size=max_input_size
    )

    model.rpn.anchor_generator = anchor_generator
    model.rpn.head = RPNHead(256, anchor_generator.num_anchors_per_location()[0])
    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)

    return model
