from functools import partial
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

from agora.training.base_config import ResNetModelConfig

# The 3D ResNet architecture is inspired by the following sources
# 1) MedicalNet -> https://github.com/Tencent/MedicalNet/blob/master/models/resnet.py#L183
# 2) Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet -> https://github.com/kenshohara/3D-ResNets-PyTorch/blob/master/models/resnet.py#L31
# 3) Facebook research (pytorch video) -> https://github.com/facebookresearch/pytorchvideo/blob/main/pytorchvideo/models/resnet.py


def get_inplanes() -> List[int]:
    # return [64, 128, 256, 512]
    return [32, 64, 128, 256]


def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False,
    )


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    # Create zero-padding tensor on the same device and dtype as out
    zero_pads = torch.zeros(
        out.size(0),
        planes - out.size(1),
        out.size(2),
        out.size(3),
        out.size(4),
        device=out.device,
        dtype=out.dtype,
    )
    # Concatenate along channel dimension
    out = torch.cat([out, zero_pads], dim=1)
    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout3d(0.1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)  # Test dropout
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super().__init__()

        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            padding=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet3DEncoder(nn.Module):
    def __init__(
        self,
        block,
        layers,
        block_inplanes,
        n_input_channels=1,
        n_classes=1,
        classifier=False,
        dropout=0.0,
        no_max_pool=False,
        shortcut_type="B",
        no_cuda=False,
    ):
        super().__init__()

        self.in_planes = block_inplanes[0]
        self.classifier = classifier
        self.no_max_pool = no_max_pool
        self.no_cuda = no_cuda

        self.conv1 = nn.Conv3d(
            n_input_channels,
            self.in_planes,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False,
        )

        self.bn1 = nn.BatchNorm3d(block_inplanes[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(
            block, block_inplanes[0], layers[0], shortcut_type
        )
        self.layer2 = self._make_layer(
            block, block_inplanes[1], layers[1], shortcut_type, stride=2
        )
        self.layer3 = self._make_layer(
            block,
            block_inplanes[2],
            layers[2],
            shortcut_type,
            stride=1,  # dilation=2
        )
        self.layer4 = self._make_layer(
            block,
            block_inplanes[3],
            layers[3],
            shortcut_type,
            stride=1,  # dilation=4
        )

        if self.classifier:
            self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(block_inplanes[3] * block.expansion, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(256, n_classes),
            )

        self._init_weights()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == "A":
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                )
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion),
                )

        layers = []
        layers.append(
            block(
                in_planes=self.in_planes,
                planes=planes,
                stride=stride,
                downsample=downsample,
            )
        )
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.classifier:
            x = self.avgpool(x)
            # x = x.view(x.size(0), -1)
            x = self.fc(x)

        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ClassifierHead(nn.Module):
    def __init__(self, in_channels: int, n_classes: int, dropout: float = 0.0):
        super().__init__()
        # self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, 256),  # Dense layer size from paper
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResNet3D(nn.Module):
    """
    Late fusion of T2w,DWI and ADC modalities with ResNet18 Encoder
    """

    def __init__(self, model_cfg: ResNetModelConfig):
        super().__init__()

        self.t2w_block = generate_model(
            model_depth=model_cfg.model_depth, classifier=False
        )
        self.dwi_block = generate_model(
            model_depth=model_cfg.model_depth, classifier=False
        )
        self.adc_block = generate_model(
            model_depth=model_cfg.model_depth, classifier=False
        )

        in_planes = get_inplanes()

        # Feature extraction after concat of modality specific encoders
        self.features = nn.Sequential(
            nn.Conv3d(
                in_planes[-1] * 3, 512, kernel_size=1
            ),  # 768→512, no spatial change
            nn.ReLU(inplace=True),
            nn.GroupNorm(num_groups=8, num_channels=512),
            nn.Conv3d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.GroupNorm(num_groups=8, num_channels=256),
            nn.Conv3d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.GroupNorm(num_groups=8, num_channels=128),
            nn.Conv3d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.GroupNorm(num_groups=8, num_channels=64),
        )

        self.gap = nn.AdaptiveAvgPool3d(1)

        # Fusion classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(model_cfg.img_feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=model_cfg.cls_dropout_p),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=model_cfg.cls_dropout_p),
            nn.Linear(32, model_cfg.cls_num_classes),
        )

    def forward(self, x):
        N, C, D, H, W = x.shape

        t2w = x[:, 0].unsqueeze(1)
        adc = x[:, 1].unsqueeze(1)
        dwi = x[:, 2].unsqueeze(1)
        # print("x shape", x.shape)
        # print("t2w shape", t2w.shape)

        # Process each modality
        xA = self.t2w_block(t2w)
        # print("xA shape",xA.shape)
        xB = self.adc_block(adc)

        xC = self.dwi_block(dwi)

        x_cat = torch.cat([xA, xB, xC], dim=1)

        # Add channel dimension if needed
        if len(x_cat.shape) == 4:
            x_cat = x_cat.unsqueeze(1)

        img_features = self.features(x_cat)

        # Global Average pooling of spatial dimensions before classifying
        x = self.gap(img_features)

        # Pass through classifier
        x = self.classifier(x)

        return x

    def get_model_summary(self, input_shape, clinical_shape=None):
        """Get summary of model architecture and parameters."""

        return summary(
            self,
            input_size=(1, 3, *input_shape),  # e.g. *25, 150, 150
            verbose=0,
            col_names=(
                "input_size",
                "output_size",
                "num_params",
                "mult_adds",
                "trainable",
            ),
            # depth=1,
        )


class ResNet3DClinical(ResNet3D):
    """
    Late fusion of T2w,DWI and ADC modalities with ResNet18 Encoder
    """

    def __init__(self, model_cfg: ResNetModelConfig):
        super().__init__(model_cfg)

        self.clin_feature_dim = model_cfg.clin_feature_dim  # 32
        self.clin_input_dim = model_cfg.clin_input_dim  # 4
        self.img_feature_dim = model_cfg.img_feature_dim  # Fused image dimension
        self.att_fusion = model_cfg.att_fusion
        self.img_pooling = model_cfg.img_pooling

        # MLP for clinical features
        self.clinical_mlp = nn.Sequential(
            nn.Linear(self.clin_input_dim, self.clin_feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=model_cfg.cls_dropout_p),
            nn.Linear(self.clin_feature_dim, self.clin_feature_dim),
            nn.ReLU(inplace=True),
        )

        if self.att_fusion:
            # Cross-attention: clinical attends to image
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=self.img_feature_dim,
                num_heads=model_cfg.att_num_heads,
                dropout=model_cfg.att_dropout_p,
                batch_first=True,
            )

            self.clinical_projection = nn.Linear(
                self.clin_feature_dim, self.img_feature_dim
            )

            # self.image_projection = nn.Linear(
            #     self.img_feature_dim, self.clin_feature_dim
            # )

            # Layer norm after attention
            self.attention_norm = nn.LayerNorm(self.img_feature_dim)

            fuse_feature_dim = self.img_feature_dim  # self.clin_feature_dim
        else:
            fuse_feature_dim = self.clin_feature_dim + self.img_feature_dim

        self.flatten = nn.Flatten(2)

        # Fusion classifier
        self.classifier = nn.Sequential(
            nn.Linear(fuse_feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=model_cfg.cls_dropout_p),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=model_cfg.cls_dropout_p),
            nn.Linear(128, model_cfg.cls_num_classes),
        )

    def forward(self, image: torch.Tensor, clinical: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Args:
            image: (N, 3, D, H, W) - T2W, ADC, HBV channels
            clinical: (N, clinical_input_dim)

        Returns:
            logits: (N, num_classes)
        """
        N, C, D, H, W = image.shape
        t2w = image[:, 0].unsqueeze(1)
        adc = image[:, 1].unsqueeze(1)
        dwi = image[:, 2].unsqueeze(1)
        # print("x shape", x.shape)
        # print("t2w shape", t2w.shape)

        # Process each modality
        xA = self.t2w_block(t2w)
        # print("xA shape",xA.shape)
        xB = self.adc_block(adc)

        xC = self.dwi_block(dwi)

        x_cat = torch.cat([xA, xB, xC], dim=1)

        # Add channel dimension if needed
        if len(x_cat.shape) == 4:
            x_cat = x_cat.unsqueeze(1)

        # Fuse modality resnet embeddings
        img_features = self.features(x_cat)

        # Clinical features
        clin_features = self.clinical_mlp(clinical)  # (N, 32)

        # Fuse and classify
        if self.att_fusion:
            # Create (B, seq_len, embed_dim) shape for MHA
            if self.img_pooling == "gap":
                img_features = self.gap(img_features).view(N, 1, -1)  # (B, 1, 64)
            else:
                img_features = self.flatten(img_features)  # -> (B,64,D*H*W)
                img_features = img_features.permute(0, 2, 1)  # (B, D*H*W, 64)
            # Project image features to same dimension
            # img_features = self.image_projection(img_features)  # (N, fusion_hidden_dim)
            # Project clinical features to same dimension
            clin_features = self.clinical_projection(
                clin_features
            )  # (N, fusion_hidden_dim)

            # Prepare for attention (add sequence dimension)
            # Clinical is query, image is key/value
            clin_query = clin_features.unsqueeze(1)  # (N, 1, fusion_hidden_dim)
            img_kv = img_features  # (N, seq_len, embed_dim)

            # Cross-attention
            attended_features, attention_weights = self.cross_attention(
                query=clin_query,
                key=img_kv,
                value=img_kv,
            )
            attended_features = attended_features.squeeze(1)  # (N, fusion_hidden_dim)

            # Residual connection + layer norm
            fused = self.attention_norm(attended_features + clin_features)
        else:
            # Global Average pooling of spatial dimensions before classifying
            img_features = self.gap(img_features)
            img_features = img_features.view(N, -1)  # (N, 64)
            fused = torch.cat([img_features, clin_features], dim=1)  # (N, 64 + 32)

        logits = self.classifier(fused)  # (N, num_classes)

        return logits

    def get_model_summary(self, input_shape, clinical_shape=4):  # type: ignore[override]
        """Get summary of model architecture and parameters."""

        return summary(
            self,
            input_size=[
                (1, 3, *input_shape),  # e.g. *25, 150, 150
                (1, clinical_shape),  # Clinical input (B, num_features)
            ],
            verbose=0,
            col_names=(
                "input_size",
                "output_size",
                "num_params",
                "mult_adds",
                "trainable",
            ),
            # depth=1,
        )


def generate_model(model_depth, **kwargs):
    model_depth_choices = [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet3DEncoder(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet3DEncoder(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet3DEncoder(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet3DEncoder(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet3DEncoder(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet3DEncoder(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet3DEncoder(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)
    else:
        raise ValueError(f"Model depth should be in {model_depth_choices}")

    return model


def main():
    # model = generate_model(model_depth=18)

    # for name, child in model.named_children():
    #     # print(f"Name: {name}")
    #     if name == "encoder":
    #         for param in child.parameters():
    #             param.requires_grad = False

    # print(model.encoder)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = ResNetModelConfig(
        model_name="dummyname", img_feature_dim=64, att_fusion=True, use_clinical=False
    )  # Default config

    # Test ResNetClassifier
    # model = ResNet3DClinical(cfg).to(device)
    model = ResNet3D(cfg).to(device)
    model_stats = model.get_model_summary((25, 150, 150), 4)
    print(model_stats)


if __name__ == "__main__":
    main()
