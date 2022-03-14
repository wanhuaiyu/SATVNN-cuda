import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
def batch_format(dataset, T_in_seq, T_out_seq, time_major=True):
    """
        Format the dataset into the form [T_in_seq+T_out_seq, n_samples, n_dims] from the form [T, n_dims]
        :param dataset: The dataset in the form  [T, n_dims]
        :param T_in_seq: Model input sequence length
        :param T_out_seq: Model output sequence length
        :param time_major: True if the results are sent in the form [T_in_seq+T_out_seq, n_samples, n_inputs].
        :return: inputs: The inputs in the form [T_in_seq+T_out_seq, n_samples, n_dims]
        :return: outputs: The inputs in the form [T_in_seq+T_out_seq, n_samples, n_dims]
        """
    T, n_dims = dataset.shape
    inputs = []
    targets = []
    # Loop over the indexes, extract a sample at that index and run it through the model
    for t in range(T - T_in_seq - T_out_seq + 1):
        # Extract the training and testing samples at the current permuted index
        inputs.append(dataset[t:t + T_in_seq, :])
        targets.append(dataset[t + T_in_seq:t + T_in_seq + T_out_seq, :])
    # Convert lists to arrays of size [n_samples, T_in, N] and [n_samples, T_out, N]
    inputs = np.array(inputs)
    targets = np.array(targets)

    if time_major:
        inputs = np.transpose(inputs, (1, 0, 2))
        targets = np.transpose(targets, (1, 0, 2))

    return inputs, targets

def generate_data(data,period):
    """
        Generate a dataset，returns a training, test, and validation dataset
        :param data: all data
        :param period: The period of the time-series seasonal component
        :return train_data: the dataset for training the model
        :return test_data: the dataset for testing the model
        :return valid_data: the dataset for validating the model.
        :return period: The period of the fundamental seasonal component of the time series.
        :return mm: Data Normalization
        """

    T_in_seq = 2 * period
    T_out_seq = period
    dataset=data.reshape(-1,1)

    n_samples = len(dataset) - T_in_seq - T_out_seq + 1
    test_idx = n_samples - int(0.2 * n_samples)
    valid_idx = n_samples - int(0.1 * n_samples)

    mm = MinMaxScaler()
    dataset = mm.fit_transform(dataset)

    inputs, targets = batch_format(dataset, T_in_seq, T_out_seq, time_major=True)

    train_x = inputs[:, :test_idx, :]
    train_y = targets[:, :test_idx, :]
    test_x = inputs[:, test_idx:valid_idx, :]
    test_y = targets[:, test_idx:valid_idx, :]
    valid_x = inputs[:, valid_idx:, :]
    valid_y = targets[:, valid_idx:, :]

    return train_x, train_y, test_x, test_y, valid_x, valid_y, period, mm

def format_input(input):
    """
    Format the input array by combining the time and input dimension of the input for feeding into model.
    :param input: Input tensor with shape [in_seq_length + out_seq_length, n_samples, input_dim]
    :return: input tensor reshaped to [n_samples, in_seq_length + out_seq_length]
    """
    in_seq_length, batch_size, input_dim = input.shape
    input_reshaped = input.permute(1, 0, 2)
    input_reshaped = torch.reshape(input_reshaped, (batch_size, -1))
    return input_reshaped