from dataclasses import fields, is_dataclass

import yaml


def flatten_config(obj, prefix="") -> dict:
    """Recursively flatten a nested dataclass into a dot-notation dict for mlflow.log_params.

    mlflow.log_params expects a flat {str: str} dict, but our configs are nested dataclasses
    (e.g. BaseConfig contains BaseTrainingConfig and BaseModelConfig). This walks the fields
    recursively and produces keys like 'training.learning_rate' or 'model.model_name'.
    """
    result = {}
    for f in fields(obj):
        val = getattr(obj, f.name)
        key = f"{prefix}.{f.name}" if prefix else f.name
        if is_dataclass(val):
            result.update(flatten_config(val, key))
        else:
            result[key] = str(val)
    return result


def load_config(config_path: str) -> dict:
    """Load and validate configuration from YAML file."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        required_keys = {"model", "training"}
        if not all(key in config for key in required_keys):
            raise ValueError(
                f"Configuration must contain all required keys: {required_keys}"
            )

        return config
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")
    except Exception as e:
        raise ValueError(f"Error loading configuration: {e}")
