import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

from agora.training.base_config import OmniRadModelConfig


class OmniRadProstateCsPCa(nn.Module):
    """
    OmniRad framework based classifier. This class supports the following classification backbones:
    ViT, DINOv2, DINOv3, CLIP, RadioDino. The trained backbones are available on HuggingFace.

    Args:
        nn (_type_): _description_
    """

    def __init__(self, model_cfg: OmniRadModelConfig):
        super().__init__()

        # Load OmniRad-base from Hugging Face Hub
        self.encoder = timm.create_model(
            model_name=model_cfg.model_name,
            pretrained=model_cfg.pretrained,
            num_classes=model_cfg.enc_num_classes,  # return embeddings if 0
        )

        self.use_dense_features = model_cfg.use_dense_features
        self.embed_dim = self.encoder.num_features  # 384/768/1024/1536
        self.img_feature_dim = model_cfg.img_feature_dim  # 256

        self.gap = nn.AdaptiveAvgPool3d(1)

        if self.use_dense_features:
            # 3D Conv to aggregate dense features across depth (Spatial aggregator)
            # Input to this block will be: (N, embed_dim, D, H_p, W_p)
            self.aggregator = nn.Sequential(
                nn.Conv3d(
                    self.embed_dim,
                    512,
                    kernel_size=(3, 3, 3),
                    stride=(1, 2, 2),
                    padding=(1, 1, 1),
                ),
                nn.BatchNorm3d(512),
                nn.ReLU(inplace=True),
                nn.Dropout3d(p=model_cfg.cls_dropout_p),
                nn.Conv3d(
                    512,
                    self.img_feature_dim,
                    kernel_size=(3, 3, 3),
                    stride=(2, 2, 2),
                    padding=(1, 1, 1),
                ),
                nn.BatchNorm3d(self.img_feature_dim),
                nn.ReLU(inplace=True),
            )
        else:
            # Just use CLS token, simpler aggregation (depth aggregator)
            # self.aggregator = nn.Sequential(
            #     nn.Linear(self.embed_dim, self.img_feature_dim),
            #     nn.ReLU(inplace=True),
            #     nn.Dropout(p=model_cfg.cls_dropout_p),
            # )
            self.aggregator = nn.Sequential(
                nn.Linear(self.embed_dim, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=model_cfg.cls_dropout_p),
                nn.Linear(512, self.img_feature_dim),
                nn.ReLU(inplace=True),
            )

        # Fusion classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.img_feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=model_cfg.cls_dropout_p),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=model_cfg.cls_dropout_p),
            nn.Linear(128, model_cfg.cls_num_classes),
        )

    def extract_omnirad_features(self, image_2d: torch.Tensor) -> torch.Tensor:
        """
        Extract features from OmniRad encoder for a single 2D image batch.

        Args:
            image_2d: (N, 3, H, W), float32 in [0, 1] (assumed)

        Returns:
            If use_dense_features:
                patch_tokens: (N, num_tokens, embed_dim)  (CLS removed if present)
            Else:
                pooled_features: (N, embed_dim)
        """

        # Resize to 224x224 (DINOv3 standard)
        img_resized = F.interpolate(
            image_2d, size=(224, 224), mode="bilinear", align_corners=False
        )
        # Forward pass through encoder
        # features = self.encoder(img_resized)
        features = self.encoder.forward_features(img_resized)

        if self.use_dense_features:
            # Check if features are in (N, seq_len, embed_dim) format
            if len(features.shape) == 3:
                # Assume first token might be CLS; drop it to keep patch tokens only
                patch_tokens = features[:, 1:, :]  # (N, num_tokens, embed_dim)
                return patch_tokens
            else:
                # If features are already spatial (N, C, H, W), return as is
                return features
        else:
            # Use global pooling or CLS token
            if len(features.shape) == 3:
                # CLS token at index 0
                pooled = features[:, 0, :]  # (N, embed_dim)
            else:
                # Spatial features - apply global average pooling
                pooled = features.mean(dim=(-2, -1))  # (N, embed_dim)
            return pooled

    def extract_omnirad_features_vec(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract features from OmniRad encoder for a 3D image volume.
        Args:
            image: (N, C, D, H, W)
        Returns:
            If use_dense_features:
                patch_tokens: (N, D, num_tokens, embed_dim)
            Else:
                pooled_features: (N, D, embed_dim)
        """
        N, C, D, H, W = image.shape

        # Merge N and D: (N*D, C, H, W)
        image_2d = image.permute(0, 2, 1, 3, 4).reshape(N * D, C, H, W)

        img_resized = F.interpolate(
            image_2d, size=(224, 224), mode="bilinear", align_corners=False
        )

        features = self.encoder.forward_features(img_resized)

        if self.use_dense_features:
            if len(features.shape) == 3:
                # (N*D, seq_len, embed_dim) -> drop CLS -> (N*D, num_tokens, embed_dim)
                patch_tokens = features[:, 1:, :]
                num_tokens, embed_dim = patch_tokens.shape[1], patch_tokens.shape[2]

                # Infer spatial grid shape, matching the original forward logic
                h_patches = int(num_tokens**0.5)
                while h_patches > 1 and num_tokens % h_patches != 0:
                    h_patches -= 1
                w_patches = num_tokens // h_patches

                # (N*D, num_tokens, embed_dim) -> (N*D, embed_dim, H_p, W_p)
                spatial = patch_tokens.transpose(1, 2).reshape(
                    N * D, embed_dim, h_patches, w_patches
                )
            else:
                # Already spatial: (N*D, embed_dim, H_p, W_p)
                _, embed_dim, h_patches, w_patches = features.shape
                spatial = features

            # Reshape back: (N*D, embed_dim, H_p, W_p) -> (N, D, embed_dim, H_p, W_p)
            # then permute to (N, embed_dim, D, H_p, W_p) to match original stack output
            return (
                spatial.reshape(N, D, embed_dim, h_patches, w_patches)
                .permute(0, 2, 1, 3, 4)  # (N, embed_dim, D, H_p, W_p)
                .contiguous()
            )

        else:
            if len(features.shape) == 3:
                pooled = features[:, 0, :]  # (N*D, embed_dim)
            else:
                pooled = features.mean(dim=(-2, -1))  # (N*D, embed_dim)

            embed_dim = pooled.shape[1]
            # (N*D, embed_dim) -> (N, D, embed_dim) -> (N, embed_dim, D)
            # permute to match original torch.stack(slice_features, dim=2) output
            return pooled.reshape(N, D, embed_dim).permute(0, 2, 1).contiguous()

    def forward(self, image: torch.Tensor) -> torch.Tensor:  # , clinical: torch.Tensor
        """
        Args:
            image: (N, 3, D, H, W) - T2W, ADC, HBV channels
            clinical: (N, clinical_input_dim)

        Returns:
            logits: (N, num_classes)
        """
        N, C, D, H, W = image.shape

        if self.use_dense_features:
            # (N, embed_dim, D, H_p, W_p) — single batched forward pass
            volume_features = self.extract_omnirad_features_vec(image)
            # Apply 3D convolutions
            volume_features = self.gap(volume_features)  # Global pooling
            img_features = self.aggregator(volume_features)  # (N, 256, 1, 1, 1)
            img_features = img_features.view(N, -1)  # (N, 256)
        else:
            # (N, embed_dim, D) — single batched forward pass
            volume_embedding = self.extract_omnirad_features_vec(image)
            # Average pool across depth slices: (N, embed_dim)
            volume_embedding = volume_embedding.mean(dim=2)
            img_features = self.aggregator(volume_embedding)  # (N, 256)

        # Classify
        logits = self.classifier(img_features)  # (N, num_classes)

        return logits

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


class OmniRadProstateCsPCaClinical(OmniRadProstateCsPCa):
    """
    Late-fusion model: combines 3D image features from a backbone (v1 or v2)
    with clinical/tabular features via a small MLP and a fusion classifier.
    """

    def __init__(self, model_cfg):
        super().__init__(model_cfg)

        self.clin_feature_dim = model_cfg.clin_feature_dim  # 32
        self.clin_input_dim = model_cfg.clin_input_dim  # 4
        self.att_fusion = model_cfg.att_fusion

        # MLP for clinical features
        self.clinical_mlp = nn.Sequential(
            nn.Linear(self.clin_input_dim, self.clin_feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=model_cfg.cls_dropout_p),
            nn.Linear(self.clin_feature_dim, self.clin_feature_dim),
            nn.ReLU(inplace=True),
        )

        if self.use_dense_features:
            if model_cfg.img_pooling == "gap":
                self.aggregator.append(nn.AdaptiveAvgPool3d((1, 1, 1)))  # -> Bx256x1x1x1
            self.aggregator = self.aggregator.append(nn.Flatten(2)) # -> Bx256xD*H*W

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

        if self.use_dense_features:
            # (N, embed_dim, D, H_p, W_p) — single batched forward pass
            volume_features = self.extract_omnirad_features_vec(image)
            # Apply 3D convolutions
            # img_features = self.aggregator(volume_features)  # (N, 256, 1, 1, 1)
            # img_features = img_features.view(N, -1)  # (N, 256)
            img_features = self.aggregator(volume_features)  # -> (N,256,D*H*W)
            img_features = img_features.permute(0, 2, 1)  # (N, D*H*W, 64)
        else:
            # (N, embed_dim, D) — single batched forward pass
            volume_embedding = self.extract_omnirad_features_vec(image)
            # Average pool across depth slices: (N, embed_dim)
            volume_embedding = volume_embedding.mean(dim=2)
            img_features = self.aggregator(volume_embedding)  # (N, 256)

        # Clinical features
        clin_features = self.clinical_mlp(clinical)  # (N, 32)

        # Fuse and classify
        if self.att_fusion:
            # Project image features to same dimension
            # img_features = self.image_projection(img_features)  # (N, fusion_hidden_dim)
            # Project clinical features to same dimension
            clin_features = self.clinical_projection(
                clin_features
            )  # (N, fusion_hidden_dim)

            # Prepare for attention (add sequence dimension)
            # Clinical is query, image is key/value
            clin_query = clin_features.unsqueeze(1)  # (N, 1, fusion_hidden_dim)
            # img_kv = img_features.unsqueeze(1)  # (N, 1, fusion_hidden_dim)
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
            fused = torch.cat([img_features, clin_features], dim=1)  # (N, 256 + 32)

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


def main():
    cfg = OmniRadModelConfig(
        model_name="hf_hub:Snarcy/OmniRad-base",
        att_fusion=True,
        use_dense_features=True,
        use_clinical=False
    )  # Default config

    model = OmniRadProstateCsPCa(cfg)
    # model = OmniRadProstateCsPCaClinical(cfg)

    # for name, child in model.named_children():
    #     print(f"Name: {name}")
    #     if name == "encoder":
    #         for param in child.parameters():
    #             param.requires_grad = False

    print(model.encoder)

    model_stats = model.get_model_summary((25, 150, 150), 4)
    print(model_stats)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # x = torch.randn(8, 3, 28, 320, 320).to(device)
    # cls = model(x)
    # print(cls)

    # x = torch.randn(8, 1, 320, 320)
    # x_rgb = torch.cat((x, x, x), dim=1)  # .to(device)
    # # enc = model.encoder(x_rgb)
    # summary(model.encoder, input_data=x_rgb)


if __name__ == "__main__":
    main()
