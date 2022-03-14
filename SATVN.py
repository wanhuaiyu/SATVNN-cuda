import torch
import torch.nn as nn
import torch.nn.functional as F
from Self_attention_block import Self_attention_block

class SATVN(nn.Module):
    """
    Class for the Self-Attention based Time-Variant model (SATVN model)
    """
    def __init__(self, input_dim, d_model, hidden_dim, h, M, output_dim, in_seq_length, out_seq_length, device):
        """
        :param input_dim: Dimension of the inputs
        :param d_model: Dimension of query matrix
        :param hidden_dim: Number of hidden units
        :param h: Number of heads of Truncated Cauchy self-attention
        :param M: Number of encoder layers of SATVN model
        :param output_dim: Dimension of the outputs
        :param in_seq_length: Length of the input sequence
        :param out_seq_length: Length of the output sequence
        """
        super(SATVN, self).__init__()

        self.input_dim = input_dim
        self.d_model=d_model
        self.hidden_dim = hidden_dim
        self.h = h
        self.M = M
        self.output_dim = output_dim
        self.in_seq_length = in_seq_length
        self.out_seq_length = out_seq_length
        self.device = device

        # Input dimension of componed inputs and sequences
        input_dim_comb = input_dim * in_seq_length


        # Initialise layers
        hidden_layer1 = [nn.Linear(input_dim_comb, hidden_dim)]
        for i in range(out_seq_length - 1):
            hidden_layer1.append(nn.Linear(input_dim_comb + hidden_dim + output_dim,  hidden_dim))
        self.hidden_layer1 = nn.ModuleList(hidden_layer1)
        self.hidden_layer2 = nn.ModuleList(
            [Self_attention_block(input_dim, d_model, hidden_dim, output_dim, d_model, d_model, h, M, attention_size=12, dropout=0.3, chunk_mode=None,
                         pe='regular') for i in range(out_seq_length)])
        self.Linear_layer = nn.ModuleList([nn.Linear(hidden_dim, output_dim) for i in range(out_seq_length)])


    def forward(self, input, target, is_training=False):
        """
        Forward propagation of the SATVN model
        :param input: Input data in the form [n_samples, input_seq_length]
        :param target: Target data in the form [output_seq_length, n_samples, output_dim]
        :param is_training: If true, use target data for training, else use the previous output.
        :return: outputs: Forecast outputs in the form [out_seq_length, n_samples, input_dim]
        """
        # Initialise outputs
        outputs = torch.zeros((self.out_seq_length, input.shape[0], self.output_dim)).to(self.device)
        # First input
        next_cell_input = input
        for i in range(self.out_seq_length):

            hidden = F.relu(self.hidden_layer1[i](next_cell_input)) #Format the dataset into the form [batch_size,H] from the form [batch_size,in_seq_length]
            hidden = self.hidden_layer2[i](hidden)   # Propagate through Cauchy self-attention block
            hidden = hidden.reshape((input.shape[0], -1))
            # Calculate the output （Linear layer of Cauchy self-attention block described in the paper）
            output = self.Linear_layer[i](hidden)
            outputs[i,:,:] = output
            # Prepare the next input
            if is_training:
                next_cell_input = torch.cat((input, hidden, target[i, :, :]), dim=1)
            else:
                next_cell_input = torch.cat((input, hidden, outputs[i, :, :]), dim=1)
        return outputs


