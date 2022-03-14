"""
Code to Evaluate of using the Mean Absolute Scaled Error (MASE) and the Symmetric Mean Absolute Percentage Error (SMAPE) for a given test set.

"""

import numpy as np
import torch
from dataHelpers import format_input
from calculateError import calculate_error

def evaluate(SA, test_x, test_y, return_lists=False):
    """
    Calculate various error metrics on a test dataset
    :param SA: A SATVNN object defined by the class in SATVNN.py
    :param test_x: Input test data in the form [in_seq_length+out_seq_length, n_batches, input_dim]
    :param test_y: target data in the form [in_seq_length+out_seq_length, n_batches, input_dim]
    :return: mase: Mean absolute scaled error
    :return: smape: Symmetric absolute percentage error
    :return: nrmse: Normalised root mean squared error
    """
    SA.model.eval()
    predict_start = 24

    # Load model parameters
    checkpoint = torch.load(SA.save_file, map_location=SA.device)
    SA.model.load_state_dict(checkpoint['model_state_dict'])
    SA.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    with torch.no_grad():
        if type(test_x) is np.ndarray:
            test_x = torch.from_numpy(test_x).type(torch.FloatTensor)
        if type(test_y) is np.ndarray:
            test_y = torch.from_numpy(test_y).type(torch.FloatTensor)

        # Format the inputs
        test_x = format_input(test_x)

        # Send to CPU/GPU
        test_x = test_x.to(SA.device)
        test_y = test_y.to(SA.device)

        # Number of batch samples
        n_samples = test_x.shape[0]

        # Inference
        y_pred_list = []
        # Compute outputs for a mixture density network output
        # Compute outputs for a linear output
        y_pred = SA.model(test_x, test_y, is_training=False)

        mase_list = []
        smape_list = []
        nrmse_list = []
        for i in range(n_samples):
            mase, se, smape, nrmse = calculate_error(y_pred[:, i, :].cpu().numpy(), test_y[:, i,
                                                                                    :].cpu().numpy())  # y_pred 和test_y 最后的shape是什么样子的
            mase_list.append(mase)
            smape_list.append(smape)
            nrmse_list.append(nrmse)
        # writer.close()
        mase = np.mean(mase_list)
        smape = np.mean(smape_list)
        nrmse = np.mean(nrmse_list)

    if return_lists:
        return np.ndarray.flatten(np.array(mase_list)), np.ndarray.flatten(np.array(smape_list)), np.ndarray.flatten(
            np.array(nrmse_list))
    else:
        return mase, smape, nrmse
