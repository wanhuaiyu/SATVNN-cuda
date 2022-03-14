import torch
def mse_loss(input, target):
    """
    Calculate the mean squared error loss
    :param input: Input sample
    :param target: Target sample
    :return mse: The mean squared error
    """
    mse = torch.mean((input - target) ** 2)
    return mse