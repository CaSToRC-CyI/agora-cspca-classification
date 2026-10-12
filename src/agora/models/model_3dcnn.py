import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

from agora.training.base_config import CNNModelConfig


class CNN3D(nn.Module):
    def __init__(self, config: CNNModelConfig):
        """Initialize 3D CNN model following paper's filter structure with original input size.

        Args:
            config: Model configuration
            num_classes: Number of output classes (default 1 for binary classification)
        """
        super().__init__()
        self.config = config

        # Feature extraction BEFORE CONCAT following paper's filter structure

        self.single_channel_block_t2w = nn.Sequential(
            # First block: 25x128x128 -> 25x128x128
            nn.Conv3d(
                in_channels=1,
                out_channels=24,  # First filter count from paper
                kernel_size=3,
                padding="same",
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),  # Output: 12x64x64
            nn.GroupNorm(num_groups=8, num_channels=24),
        )

        self.single_channel_block_adc = nn.Sequential(
            # First block: 25x128x128 -> 25x128x128
            nn.Conv3d(
                in_channels=1,
                out_channels=24,  # First filter count from paper
                kernel_size=3,
                padding="same",
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),  # Output: 12x64x64
            nn.GroupNorm(num_groups=8, num_channels=24),
        )

        self.single_channel_block_dwi = nn.Sequential(
            # First block: 25x128x128 -> 25x128x128
            nn.Conv3d(
                in_channels=1,
                out_channels=24,  # First filter count from paper
                kernel_size=3,
                padding="same",
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),  # Output: 12x64x64
            nn.GroupNorm(num_groups=8, num_channels=24),
        )

        # Feature extraction AFTER CONCAT layers following paper's filter structure
        self.features = nn.Sequential(
            # Second block
            nn.Conv3d(
                in_channels=72,
                out_channels=96,  # Second filter count from paper
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),  # Output: 12x64x64
            nn.GroupNorm(num_groups=8, num_channels=96),
            # Third block
            nn.Conv3d(
                in_channels=96,
                out_channels=192,  # Third filter count from paper
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),  # Output: 12x64x64
            nn.GroupNorm(num_groups=8, num_channels=192),
            # Fourth block
            nn.Conv3d(
                in_channels=192,
                out_channels=320,  # Fourth filter count from paper
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),  # Output: 12x64x64
            nn.GroupNorm(num_groups=8, num_channels=320),
        )

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool3d(1)  # Output: 320x1x1x1

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.config.img_feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.config.cls_dropout_p),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.config.cls_dropout_p),
            nn.Linear(128, self.config.cls_num_classes),
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, channels, depth, height, width)

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """

        t2w = x[:, 0].unsqueeze(1)
        adc = x[:, 1].unsqueeze(1)
        dwi = x[:, 2].unsqueeze(1)
        # print("x shape", x.shape)
        # print("t2w shape", t2w.shape)

        # Process each modality
        xA = self.single_channel_block_t2w(t2w)
        # print("xA shape",xA.shape)
        xB = self.single_channel_block_adc(adc)

        xC = self.single_channel_block_dwi(dwi)

        x = torch.cat([xA, xB, xC], dim=1)

        # Add channel dimension if needed
        if len(x.shape) == 4:
            x = x.unsqueeze(1)

        # Pass through feature extractor
        x = self.features(x)

        # Global Average pooling of spatial dimensions before classifying
        x = self.gap(x)

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


class CNN3DClinical(CNN3D):
    def __init__(self, model_cfg: CNNModelConfig):
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
        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))  # Global pooling

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

        self._initialize_weights()

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
        # print("image shape", image.shape)
        # print("t2w shape", t2w.shape)

        # Process each modality
        xA = self.single_channel_block_t2w(t2w)
        # print("xA shape",xA.shape)
        xB = self.single_channel_block_adc(adc)

        xC = self.single_channel_block_dwi(dwi)

        x_cat = torch.cat([xA, xB, xC], dim=1)

        # Add channel dimension if needed
        if len(x_cat.shape) == 4:
            x_cat = x_cat.unsqueeze(1)

        # Pass through feature extractor
        img_features = self.features(x_cat)  # -> (B,320,D,H,W)

        # Clinical features
        clin_features = self.clinical_mlp(clinical)  # (N, 32)

        # Fuse and classify
        if self.att_fusion:
            # Create (B, seq_len, embed_dim) shape for MHA
            if self.img_pooling == "gap":
                img_features = self.gap(img_features).view(N, 1, -1)  # (B, 1, 320)
            else:
                img_features = self.flatten(img_features)  # -> (B,320,D*H*W)
                img_features = img_features.permute(0, 2, 1)  # (B, D*H*W, 320)
            # Project image features to same dimension
            # img_features = self.image_projection(img_features)  # (N, fusion_hidden_dim)
            # Project clinical features to same dimension
            clin_features = self.clinical_projection(
                clin_features
            )  # (N, fusion_hidden_dim)

            # Prepare for attention (add sequence dimension)
            # Clinical is query, image is key/value
            clin_query = clin_features.unsqueeze(1)  # (N, 1, fusion_hidden_dim)
            img_kv = img_features

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
            img_features = img_features.view(N, -1)  # (N, 320)

            fused = torch.cat([img_features, clin_features], dim=1)  # (N, 320 + 32)

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


FEAT_EXT_FILT = 24
FEAT_COMB_FILT_1 = 72
POOL_SIZES = [1, 2, 4]


class SpatialPyramidPooling3D(nn.Module):
    def __init__(self, pool_sizes=[1, 2, 4]):
        super().__init__()
        self.pool_sizes = pool_sizes

    def forward(self, x):
        # x: [batch, channels, depth, height, width]
        batch_size, _channels = x.size(0), x.size(1)
        pooled_features = []

        for pool_size in self.pool_sizes:
            # Pool to [batch, channels, pool_size, pool_size, pool_size]
            # nn.AdaptiveMaxPool3d
            pooled = F.adaptive_max_pool3d(x, output_size=pool_size)

            # Flatten to [batch, channels * pool_size^3]
            flattened = pooled.view(batch_size, -1)

            pooled_features.append(flattened)

        # Concatenate all flattened vectors: [batch, total_features]
        return torch.cat(pooled_features, dim=1)


class CNN3D_Mod(CNN3D):
    def __init__(self, config: CNNModelConfig, num_classes=1):
        """Initialize Simplified 3D CNN model following paper's filter structure with original input size.

        Args:
            config: Model configuration
            num_classes: Number of output classes (default 1 for binary classification)
        """
        # super().__init__()
        nn.Module.__init__(self)
        self.config = config

        # Feature extraction BEFORE CONCAT following paper's filter structure

        self.single_channel_block_t2w = nn.Sequential(
            # First block: 25x128x128 -> 25x128x128
            nn.Conv3d(
                in_channels=1,
                out_channels=FEAT_EXT_FILT,  # First filter count from paper
                kernel_size=3,
                padding="same",
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),  # Output: 12x64x64
            nn.GroupNorm(num_groups=8, num_channels=FEAT_EXT_FILT),
        )

        self.single_channel_block_adc = nn.Sequential(
            # First block: 25x128x128 -> 25x128x128
            nn.Conv3d(
                in_channels=1,
                out_channels=FEAT_EXT_FILT,  # First filter count from paper
                kernel_size=3,
                padding="same",
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),  # Output: 12x64x64
            nn.GroupNorm(num_groups=8, num_channels=FEAT_EXT_FILT),
        )

        self.single_channel_block_dwi = nn.Sequential(
            # First block: 25x128x128 -> 25x128x128
            nn.Conv3d(
                in_channels=1,
                out_channels=FEAT_EXT_FILT,  # First filter count from paper
                kernel_size=3,
                padding="same",
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),  # Output: 12x64x64
            nn.GroupNorm(num_groups=8, num_channels=FEAT_EXT_FILT),
        )

        # Feature extraction AFTER CONCAT layers following paper's filter structure
        self.features = nn.Sequential(
            # Second block
            nn.Conv3d(
                in_channels=FEAT_EXT_FILT * 3,
                out_channels=FEAT_COMB_FILT_1,  # Second filter count from paper
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),  # Output: 12x64x64
            nn.GroupNorm(num_groups=8, num_channels=FEAT_COMB_FILT_1),
            # nn.Dropout(p=0.2),
            # Third block
            nn.Conv3d(
                in_channels=FEAT_COMB_FILT_1,
                out_channels=FEAT_COMB_FILT_1 * 2,  # Third filter count from paper
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),  # Output: 12x64x64
            nn.GroupNorm(num_groups=8, num_channels=FEAT_COMB_FILT_1 * 2),
            # nn.Dropout(p=0.2),
            # Fourth block
            nn.Conv3d(
                in_channels=FEAT_COMB_FILT_1 * 2,
                out_channels=FEAT_COMB_FILT_1 * 4,  # Fourth filter count from paper
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),  # Output: 12x64x64
            nn.GroupNorm(num_groups=8, num_channels=FEAT_COMB_FILT_1 * 4),
            # nn.Dropout(p=0.2),
            # Global Average Pooling
            nn.AdaptiveAvgPool3d(1),  # Output: 320x1x1x1
            # SpatialPyramidPooling3D(pool_sizes=POOL_SIZES),
        )

        # self.classifier = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(640, 960),  # Dense layer size from paper
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=self.config.cls_dropout_p),
        #     nn.Linear(960, 480),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=self.config.cls_dropout_p),
        #     nn.Linear(480, 240),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=self.config.cls_dropout_p),
        #     nn.Linear(240, num_classes),
        # )

        # Classifier layers following paper structure
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(FEAT_COMB_FILT_1 * 4, 240),  # Dense layer size from paper
            # nn.Linear(FEAT_COMB_FILT_1 * 4 * 73, 240),  # For SPP
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.config.cls_dropout_p),
            nn.Linear(240, num_classes),
        )

        # Initialize weights
        self._initialize_weights()


class CNN3D_Simple(CNN3D):
    def __init__(self, config: CNNModelConfig, num_classes=1):
        """Initialize Simplified 3D CNN model following paper's filter structure with original input size.

        Args:
            config: Model configuration
            num_classes: Number of output classes (default 1 for binary classification)
        """
        # super().__init__()
        nn.Module.__init__(self)
        self.config = config

        # Feature extraction BEFORE CONCAT following paper's filter structure

        self.single_channel_block_t2w = nn.Sequential(
            # First block: 25x128x128 -> 25x128x128
            nn.Conv3d(
                in_channels=1,
                out_channels=12,  # First filter count from paper
                kernel_size=3,
                padding="same",
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),  # Output: 12x64x64
            nn.GroupNorm(num_groups=4, num_channels=12),
        )

        self.single_channel_block_adc = nn.Sequential(
            # First block: 25x128x128 -> 25x128x128
            nn.Conv3d(
                in_channels=1,
                out_channels=12,  # First filter count from paper
                kernel_size=3,
                padding="same",
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),  # Output: 12x64x64
            nn.GroupNorm(num_groups=4, num_channels=12),
        )

        self.single_channel_block_dwi = nn.Sequential(
            # First block: 25x128x128 -> 25x128x128
            nn.Conv3d(
                in_channels=1,
                out_channels=12,  # First filter count from paper
                kernel_size=3,
                padding="same",
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),  # Output: 12x64x64
            nn.GroupNorm(num_groups=4, num_channels=12),
        )

        # Feature extraction AFTER CONCAT layers following paper's filter structure
        self.features = nn.Sequential(
            # Second block
            nn.Conv3d(
                in_channels=36,
                out_channels=48,  # Second filter count from paper
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),  # Output: 12x64x64
            nn.GroupNorm(num_groups=8, num_channels=48),
            # nn.Dropout(p=0.2),
            # Third block
            nn.Conv3d(
                in_channels=48,
                out_channels=96,  # Third filter count from paper
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),  # Output: 12x64x64
            nn.GroupNorm(num_groups=8, num_channels=96),
            # nn.Dropout(p=0.2),
            # Fourth block
            nn.Conv3d(
                in_channels=96,
                out_channels=160,  # Fourth filter count from paper
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),  # Output: 12x64x64
            nn.GroupNorm(num_groups=8, num_channels=160),
            # nn.Dropout(p=0.2),
            # Global Average Pooling
            nn.AdaptiveAvgPool3d(1),  # Output: 320x1x1x1
        )

        # self.classifier = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(640, 960),  # Dense layer size from paper
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=self.config.cls_dropout_p),
        #     nn.Linear(960, 480),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=self.config.cls_dropout_p),
        #     nn.Linear(480, 240),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=self.config.cls_dropout_p),
        #     nn.Linear(240, num_classes),
        # )

        # Classifier layers following paper structure
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(160, 240),  # Dense layer size from paper
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.config.cls_dropout_p),
            nn.Linear(240, num_classes),
        )

        # Initialize weights
        self._initialize_weights()


class CNN3D_Large(CNN3D):
    def __init__(self, config: CNNModelConfig, num_classes=1):
        """Initialize Simplified 3D CNN model following paper's filter structure with original input size.

        Args:
            config: Model configuration
            num_classes: Number of output classes (default 1 for binary classification)
        """
        # super().__init__()
        nn.Module.__init__(self)
        self.config = config

        # Feature extraction BEFORE CONCAT following paper's filter structure

        self.single_channel_block_t2w = nn.Sequential(
            # First block: 25x128x128 -> 25x128x128
            nn.Conv3d(
                in_channels=1,
                out_channels=48,  # First filter count from paper
                kernel_size=3,
                padding="same",
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),  # Output: 12x64x64
            nn.GroupNorm(num_groups=8, num_channels=48),
        )

        self.single_channel_block_adc = nn.Sequential(
            # First block: 25x128x128 -> 25x128x128
            nn.Conv3d(
                in_channels=1,
                out_channels=48,  # First filter count from paper
                kernel_size=3,
                padding="same",
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),  # Output: 12x64x64
            nn.GroupNorm(num_groups=8, num_channels=48),
        )

        self.single_channel_block_dwi = nn.Sequential(
            # First block: 25x128x128 -> 25x128x128
            nn.Conv3d(
                in_channels=1,
                out_channels=48,  # First filter count from paper
                kernel_size=3,
                padding="same",
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),  # Output: 12x64x64
            nn.GroupNorm(num_groups=8, num_channels=48),
        )

        # Feature extraction AFTER CONCAT layers following paper's filter structure
        self.features = nn.Sequential(
            # Second block
            nn.Conv3d(
                in_channels=144,
                out_channels=192,  # Second filter count from paper
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),  # Output: 12x64x64
            nn.GroupNorm(num_groups=8, num_channels=192),
            # nn.Dropout(p=0.2),
            # Third block
            nn.Conv3d(
                in_channels=192,
                out_channels=384,  # Third filter count from paper
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),  # Output: 12x64x64
            nn.GroupNorm(num_groups=8, num_channels=384),
            # nn.Dropout(p=0.2),
            # Fourth block
            nn.Conv3d(
                in_channels=384,
                out_channels=640,  # Fourth filter count from paper
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),  # Output: 12x64x64
            nn.GroupNorm(num_groups=8, num_channels=640),
            # nn.Dropout(p=0.2),
            # Global Average Pooling
            nn.AdaptiveAvgPool3d(1),  # Output: 320x1x1x1
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(640, 960),  # Dense layer size from paper
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.config.cls_dropout_p),
            nn.Linear(960, 480),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.config.cls_dropout_p),
            nn.Linear(480, 240),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.config.cls_dropout_p),
            nn.Linear(240, num_classes),
        )


class FCCNN3D_Simple(CNN3D_Simple):
    def __init__(self, config: CNNModelConfig, num_classes=1):
        super().__init__(config, num_classes)

        # Modify the fused feature extraction to remove Average pooling
        self.features = nn.Sequential(
            # Second block
            nn.Conv3d(
                in_channels=36,
                out_channels=48,  # Second filter count from paper
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),  # Output: 48x37x37x7
            nn.GroupNorm(num_groups=8, num_channels=48),
            # nn.Dropout(p=0.2),
            # Third block
            nn.Conv3d(
                in_channels=48,
                out_channels=96,  # Third filter count from paper
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.GroupNorm(num_groups=8, num_channels=96),
            # nn.Dropout(p=0.2),
            # Fourth block
            nn.Conv3d(
                in_channels=96,
                out_channels=160,  # Fourth filter count from paper
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.GroupNorm(num_groups=8, num_channels=160),
            # nn.Dropout(p=0.2),
            # Global Average Pooling
            # nn.AdaptiveAvgPool3d(1),  # Output: Cx1x1x1
        )

        cls_in_shape = 160 * 9 * 9 * 1
        # Modify the Classifier Head for Fully Connected variant
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cls_in_shape, 720),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.config.cls_dropout_p),
            # nn.Linear(720, 360),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p=self.config.cls_dropout_p),
            # nn.Linear(360, 180),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p=self.config.cls_dropout_p),
            nn.Linear(720, num_classes),
        )

        # Re-initialize weights for the modified classifier head
        self._initialize_weights()


class CNN3D_Tune(CNN3D):
    def __init__(self, config, num_classes=1):
        """Initialize 3D CNN model for Hyperparameter Tuning.

        Args:
            config: Model configuration
            num_classes: Number of output classes (default 1 for binary classification)
        """
        nn.Module.__init__(self)
        self.config = config

        # Get num_filters[i] for each layer from config (total 5)
        # and num_groups[i] for each group_norm from config (total 4)

        # Feature extraction BEFORE CONCAT following paper's filter structure

        sc_filters = config["num_filters"][0]
        sc_groups = config["num_groups"][0]

        self.single_channel_block_t2w = nn.Sequential(
            # First block: 25x128x128 -> 25x128x128
            nn.Conv3d(
                in_channels=1,
                out_channels=sc_filters,  # First filter count from paper
                kernel_size=3,
                padding="same",
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),  # Output: 12x64x64
            nn.GroupNorm(num_groups=sc_groups, num_channels=sc_filters),
        )

        self.single_channel_block_adc = nn.Sequential(
            # First block: 25x128x128 -> 25x128x128
            nn.Conv3d(
                in_channels=1,
                out_channels=sc_filters,  # First filter count from paper
                kernel_size=3,
                padding="same",
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),  # Output: 12x64x64
            nn.GroupNorm(num_groups=sc_groups, num_channels=sc_filters),
        )

        self.single_channel_block_dwi = nn.Sequential(
            # First block: 25x128x128 -> 25x128x128
            nn.Conv3d(
                in_channels=1,
                out_channels=sc_filters,  # First filter count from paper
                kernel_size=3,
                padding="same",
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),  # Output: 12x64x64
            nn.GroupNorm(num_groups=sc_groups, num_channels=sc_filters),
        )

        # Feature extraction AFTER CONCAT layers following paper's filter structure
        self.features = nn.Sequential(
            # Second block
            nn.Conv3d(
                in_channels=sc_filters * 3,  # 3 MRI inputs
                out_channels=config["num_filters"][1],  # Second filter count from paper
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),  # Output: 12x64x64
            nn.GroupNorm(
                num_groups=config["num_groups"][1],
                num_channels=config["num_filters"][1],
            ),
            # Third block
            nn.Conv3d(
                in_channels=config["num_filters"][1],
                out_channels=config["num_filters"][2],  # Third filter count from paper
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),  # Output: 12x64x64
            nn.GroupNorm(
                num_groups=config["num_groups"][2],
                num_channels=config["num_filters"][2],
            ),
            # Fourth block
            nn.Conv3d(
                in_channels=config["num_filters"][2],
                out_channels=config["num_filters"][3],  # Fourth filter count from paper
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),  # Output: 12x64x64
            nn.GroupNorm(
                num_groups=config["num_groups"][3],
                num_channels=config["num_filters"][3],
            ),
            # Global Average Pooling
            nn.AdaptiveAvgPool3d(1),  # Output: 320x1x1x1
        )

        # Classifier layers following paper structure
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                config["num_filters"][3], config["num_filters"][4]
            ),  # Dense layer size from paper
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(240, 1),
        )

        # Initialize weights
        self._initialize_weights()


if __name__ == "__main__":
    # Test the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = CNNModelConfig(
        model_name="dummyname", img_feature_dim=320, att_fusion=True, use_clinical=False
    )
    # model = CNN3DClinical(config).to(device)
    model = CNN3D(config).to(device)
    # print(model.features[8])

    # Print model architecture
    model_stats = model.get_model_summary((25, 150, 150), 4)
    print(model_stats)
    # print("test")
