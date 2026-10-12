"""
Removed legacy / duplicate fields for review:

BaseModelConfig
---------------
input_shape
dropout_rate
input_size
enc_num_classes
use_dense_features

BaseTrainingConfig
------------------
pretrain_checkpoint_dir
finetune_lr
finetune_wd
layer_decay
finetune
finetune_t
layer_decay
layer-wise finetuning params

CNN / OmniRad duplicates removed
--------------------------------
data_dir
csv_file
goc_id_map
cache_dir
target_shape
output_dir
checkpoint_dir
loss_fn
optimizer
batch_size
num_epochs
learning_rate
weight_decay
num_workers
log_frequency
save_frequency
val_split
weighted_sampling
"""

import json
from dataclasses import dataclass, fields
from pathlib import Path

# ----------------- Base Model Config -----------------
from typing import Generic, Literal, Optional, Tuple, TypeVar

import yaml

from agora.models.utils import FINETUNE_MAP

TModel = TypeVar("TModel", bound="BaseModelConfig")
TTraining = TypeVar("TTraining", bound="BaseTrainingConfig")


@dataclass
class BaseModelConfig:
    """Base configuration for model architecture."""

    model_name: str
    pretrained: bool = True

    cls_num_classes: int = 1
    use_clinical: bool = True
    att_fusion: bool = False

    clin_feature_dim: int = 32
    clin_input_dim: int = 4
    img_feature_dim: int = 256

    cls_dropout_p: float = 0.2
    att_dropout_p: float = 0.2
    att_num_heads: int = 4

    model_id: str = ""

    def __post_init__(self):
        self.validate()

    def validate(self):
        """Hook for subclass validation."""
        pass


# ----------------- Base Training Config -----------------


DatasetLiteral = Literal["picai", "goc", "promis", "picai+goc", "all"]
OptimizerLiteral = Literal["adam", "adamw", "sgd", "rmsprop", "adagrad"]
LossLiteral = Literal["focal", "bce", "ce"]
PoolingLiteral = Literal["flatten", "gap"]


@dataclass
class BaseTrainingConfig:
    """Common training configuration shared across models."""

    data_dir: Path
    csv_file: Path
    goc_id_map: Path
    cache_dir: Path
    target_shape: Tuple[int, int, int]
    output_dir: Path
    checkpoint_dir: Optional[Path]

    loss_fn: LossLiteral
    optimizer: OptimizerLiteral

    datasets: DatasetLiteral = "picai"
    cspca_threshold: float = 7.0
    cache_tag: str = ""

    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    warmup_frac: float = 0.2  # fraction of num_epochs for linear LR warmup

    num_workers: int = 4
    log_frequency: int = 10
    save_frequency: int = 5

    val_split: Optional[float] = None
    weighted_sampling: bool = False

    seed: int = 42

    def __post_init__(self):
        """Convert path-like fields and run validation."""
        self.data_dir = Path(self.data_dir)
        self.csv_file = Path(self.csv_file)
        self.goc_id_map = Path(self.goc_id_map)
        self.cache_dir = Path(self.cache_dir)
        self.output_dir = Path(self.output_dir)
        self.checkpoint_dir = (
            Path(self.checkpoint_dir) if self.checkpoint_dir not in (None, "") else None
        )

        self._validate_common()
        self.validate()

    def _validate_common(self):
        """Validate shared training parameters."""

        if not self.data_dir.exists():
            raise ValueError(f"Data directory does not exist: {self.data_dir}")

        if not self.csv_file.exists():
            raise ValueError(f"CSV file does not exist: {self.csv_file}")

        if not self.goc_id_map.exists():
            raise ValueError(f"JSON file does not exist: {self.goc_id_map}")

        if self.checkpoint_dir is not None and not self.checkpoint_dir.exists():
            raise ValueError(
                f"Checkpoint directory does not exist: {self.checkpoint_dir}"
            )

        if self.batch_size < 1:
            raise ValueError(f"Batch size must be positive, got {self.batch_size}")

        if self.num_epochs < 1:
            raise ValueError(
                f"Number of epochs must be positive, got {self.num_epochs}"
            )

        if not 0 < self.learning_rate < 1:
            raise ValueError(
                f"Learning rate must be between 0 and 1, got {self.learning_rate}"
            )

        if not 0 < self.warmup_frac < 1:
            raise ValueError(
                f"Warmup fraction must be in (0, 1), got {self.warmup_frac}"
            )

        if self.val_split is not None and not 0 < self.val_split < 1:
            raise ValueError(
                f"Validation split must be between 0 and 1, got {self.val_split}"
            )

    def validate(self):
        """Hook for subclass validation."""
        pass


# ----------------- Base Config -----------------


@dataclass
class BaseConfig(Generic[TModel, TTraining]):
    """Container combining model and training configurations."""

    model: TModel
    training: TTraining

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "BaseConfig":
        """Load configuration from YAML."""
        with open(yaml_path, "r") as f:
            cfg_dict = yaml.safe_load(f)

        return cls._build(cfg_dict)

    @classmethod
    def from_json(cls, path: str) -> "BaseConfig":
        """Load configuration from JSON."""
        with open(path, "r") as f:
            cfg_dict = json.load(f)

        return cls._build(cfg_dict)

    @classmethod
    def _build(cls, cfg_dict: dict) -> "BaseConfig":
        raise NotImplementedError


# ----------------- OmniRad Config -----------------


@dataclass
class OmniRadModelConfig(BaseModelConfig):
    """Configuration for OmniRad models."""

    enc_num_classes: int = 0
    use_dense_features: bool = True
    img_pooling: PoolingLiteral = "flatten"  # att_fusion image tokens: flatten | gap

    def validate(self):
        model_name_opt = [
            "hf_hub:Snarcy/RadioDino-s16",
            "hf_hub:Snarcy/RadioDino-b16",
            "hf_hub:Snarcy/OmniRad-small",
            "hf_hub:Snarcy/OmniRad-base",
        ]

        if self.model_name not in model_name_opt:
            raise ValueError(f"model_name must be one of {model_name_opt}")


@dataclass
class OmniRadTrainingConfig(BaseTrainingConfig):
    """Training configuration specific to OmniRad."""

    finetune_lr: float = 0.0001
    finetune_wd: float = 0.00001
    layer_decay: float = 0.75

    finetune: bool = False
    finetune_t: str = "v1"

    def validate(self):
        if not 0 < self.finetune_lr < 1:
            raise ValueError(
                f"Finetune learning rate must be between 0 and 1, got {self.finetune_lr}"
            )

        if self.finetune_t not in FINETUNE_MAP and self.finetune_t != "v0":
            raise ValueError(f"Invalid finetune_type: {self.finetune_t}")


@dataclass
class OmniRadConfig(BaseConfig):
    """Full OmniRad configuration."""

    model: OmniRadModelConfig
    training: OmniRadTrainingConfig

    @classmethod
    def _build(cls, cfg_dict: dict) -> "OmniRadConfig":
        return cls(
            model=OmniRadModelConfig(**cfg_dict.get("model", {})),
            training=OmniRadTrainingConfig(**cfg_dict.get("training", {})),
        )


# ----------------- CNN Config -----------------


@dataclass
class CNNModelConfig(BaseModelConfig):
    """Configuration for simple 3D CNN models."""

    model_v: str = "v0"
    img_pooling: PoolingLiteral = "flatten"  # att_fusion image tokens: flatten | gap

    def validate(self):
        pass
        # Update in the future if more model versions are added
        # if self.model_v not in model_version_map and self.model_v != "v0":
        # raise ValueError(f"Invalid model version: {self.model_v}")


@dataclass
class CNNTrainingConfig(BaseTrainingConfig):
    """Training configuration for CNN models."""

    pass


@dataclass
class CNNConfig(BaseConfig):
    """Full CNN configuration."""

    model: CNNModelConfig
    training: CNNTrainingConfig

    @classmethod
    def _build(cls, cfg_dict: dict) -> "CNNConfig":
        return cls(
            model=CNNModelConfig(**cfg_dict.get("model", {})),
            training=CNNTrainingConfig(**cfg_dict.get("training", {})),
        )


# ----------------- ResNet Config -----------------


@dataclass
class ResNetModelConfig(BaseModelConfig):
    """Configuration for simple 3D CNN models."""

    model_depth: int = 18
    model_v: str = "v0"  # v0 is neck style fusion of image features
    img_pooling: PoolingLiteral = "flatten"  # att_fusion image tokens: flatten | gap

    def validate(self):
        pass
        # Update in the future if more model versions are added
        # if self.model_v not in model_version_map and self.model_v != "v0":
        # raise ValueError(f"Invalid model version: {self.model_v}")


@dataclass
class ResNetTrainingConfig(BaseTrainingConfig):
    """Training configuration for CNN models."""

    pass


@dataclass
class ResNetConfig(BaseConfig):
    """Full CNN configuration."""

    model: ResNetModelConfig
    training: ResNetTrainingConfig

    @classmethod
    def _build(cls, cfg_dict: dict) -> "ResNetConfig":
        return cls(
            model=ResNetModelConfig(**cfg_dict.get("model", {})),
            training=ResNetTrainingConfig(**cfg_dict.get("training", {})),
        )


# ----------------- External Evaluation Config -----------------


FileFormatLiteral = Literal["mha", "nii.gz"]


@dataclass
class ExternalEvalConfig:
    """Configuration describing an external dataset to evaluate a trained model on.

    This is intentionally decoupled from the training config: it only carries
    what is needed to build a single, un-split evaluation dataset. The model
    architecture / clinical flags are taken from the trained run's config, not
    from here.

    Expected on-disk layout (all paths relative to ``data_dir`` unless absolute):
        data_dir/<image_subdir>/{study_id}_<modality>.<file_format>
        data_dir/<seg_dir>/{study_id}.nii.gz      (prostate masks, required)

    Modality tokens follow the existing dataloader convention: either
    ``t2w``/``adc``/``hbv`` or the nnUNet channel suffixes ``_0000``/``_0001``/``_0003``.
    """

    name: str
    data_dir: Path
    image_subdir: str
    seg_dir: str
    csv_file: Path
    cache_dir: Path
    target_shape: Tuple[int, int, int]

    file_format: FileFormatLiteral = "nii.gz"
    # Optional JSON mapping raw study_id -> id used in the label CSV (e.g. for
    # anonymization). Leave unset when the image filenames already match the CSV.
    id_map: Optional[Path] = None
    cspca_threshold: float = 7.0
    cache_tag: str = ""
    labeled: bool = True

    batch_size: int = 4
    num_workers: int = 4

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.csv_file = Path(self.csv_file)
        self.cache_dir = Path(self.cache_dir)
        self.id_map = Path(self.id_map) if self.id_map not in (None, "") else None
        self.target_shape = tuple(self.target_shape)
        self._validate()

    def _validate(self):
        if not self.data_dir.exists():
            raise ValueError(f"External data_dir does not exist: {self.data_dir}")
        if not (self.data_dir / self.image_subdir).exists():
            raise ValueError(
                f"External image_subdir does not exist: {self.data_dir / self.image_subdir}"
            )
        if not (self.data_dir / self.seg_dir).exists():
            raise ValueError(
                f"External seg_dir does not exist: {self.data_dir / self.seg_dir}. "
                "Generate prostate segmentations first "
                "(see scripts/prepare_external_segmentations.py)."
            )
        if self.csv_file is not None and self.labeled and not self.csv_file.exists():
            raise ValueError(
                f"External csv_file does not exist: {self.csv_file} "
                "(required when labeled=true)."
            )
        if self.id_map is not None and not self.id_map.exists():
            raise ValueError(f"External id_map does not exist: {self.id_map}")
        if self.file_format not in ("mha", "nii.gz"):
            raise ValueError(
                f"file_format must be 'mha' or 'nii.gz', got {self.file_format!r}"
            )

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ExternalEvalConfig":
        """Load an external-eval config from a YAML file with an ``external_eval`` block."""
        with open(yaml_path, "r") as f:
            cfg_dict = yaml.safe_load(f)
        block = cfg_dict.get("external_eval", cfg_dict)
        # The external_eval block is shared with the step-0 prep script, which
        # adds keys (e.g. input_layout, label_mode) that this config doesn't
        # model. Keep only the recognized fields so one config drives both.
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in block.items() if k in known})


# ----------------- Config Registry -----------------

CONFIG_REGISTRY = {
    "omnirad": OmniRadConfig,
    "radiodino": OmniRadConfig,
    "cnn3d": CNNConfig,
    "resnet3d": ResNetConfig,
}
