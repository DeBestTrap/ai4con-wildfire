import torch
import torch.nn as nn


class ModelWrapper(nn.Module):
    def __init__(self, model, output_transform=None):
        """
        A wrapper for PyTorch models to standardize outputs.

        Parameters:
            model (nn.Module): The PyTorch model to wrap.
            output_transform (callable, optional): A function to transform the model's outputs.
                                                   Should accept the raw model output as input
                                                   and return the desired format.
        """
        super(ModelWrapper, self).__init__()
        self.model = model
        self.output_transform = output_transform

    def forward(self, *args, **kwargs):
        """
        Forward pass through the wrapped model.

        Parameters:
            *args: Positional arguments for the wrapped model.
            **kwargs: Keyword arguments for the wrapped model.

        Returns:
            The transformed outputs if `output_transform` is provided,
            otherwise the raw model outputs.
        """
        raw_output = self.model(*args, **kwargs)
        if self.output_transform:
            return self.output_transform(raw_output)
        return raw_output
