# %%
import yaml
import albumentations as A
from typing import Tuple


def load_transforms(config: dict) -> A.Compose:
    """
    Load the transforms dynamically from the config
    """

    transform_list = []
    for transform in config["transforms"]:
        # Dynamically get the transform class
        transform_class = getattr(A, transform["name"])
        # Instantiate the transform with parameters
        transform_instance = transform_class(**transform["params"])
        transform_list.append(transform_instance)

    # Compose the transforms
    return A.Compose(transform_list)


def load_config(config_path: str) -> Tuple[dict, A.Compose]:
    """
    Load the config file
    """

    # Load the YAML file
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    experiment_config = config["experiment_config"]
    transforms = load_transforms(config)

    return experiment_config, transforms


# Example usage
config_path = "config.yml"
transforms = load_config(config_path)
