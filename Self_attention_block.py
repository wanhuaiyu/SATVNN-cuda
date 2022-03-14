import torch.nn as nn
import torch
import numpy as np
from typing import Optional
from Encoder import Encoder
def generate_original_PE(length: int, d_model: int) -> torch.Tensor:
    """
    Generate positional encoding as described in original paper.  :class:`torch.Tensor`
    :param length: a higher space H of the length of input data.
    :param d_model: Dimension of the model vector.
    :return: PE: Tensor of shape (H, d_model).
    """
    PE = torch.zeros((length, d_model))

    pos = torch.arange(length).unsqueeze(1)
    PE[:, 0::2] = torch.sin(
        pos / torch.pow(1000, torch.arange(0, d_model, 2, dtype=torch.float32)/d_model))
    PE[:, 1::2] = torch.cos(
        pos / torch.pow(1000, torch.arange(1, d_model, 2, dtype=torch.float32)/d_model))

    return PE
def generate_regular_PE(length: int, d_model: int, period: Optional[int] = 24) -> torch.Tensor:
    """
    Generate positional encoding with a given period.
     :param length: a higher dimensional space H of the length of input data.
     :param d_model: Dimension of the model vector.
     :param period: Size of the pattern to repeat. Default is 12.
     :return: PE: Tensor of shape (H, d_model).
     """
    PE = torch.zeros((length, d_model))

    pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    PE = torch.sin(pos * 2 * np.pi / period)
    PE = PE.repeat((1, d_model))

    return PE
class Self_attention_block(nn.Module):
    """
      Contains Self_attention_block described in paper.
       :param d_input: Model input dimension.
       :param d_model:Dimension of the input vector.
       :param hidden_dim: Dimension of the hidden units
       :param d_output:Model output dimension.
       :param q: Dimension of queries and keys.
       :param v: Dimension of values.
       :param h: Number of heads.
       :param M: Number of encoder layers to stack
       :parama attention_size: Number of backward elements to apply attention
       :parama dropout:Dropout probability after each MHA or PFF block.Default is ``0.3``
       :parama chunk_mode:Swict between different MultiHeadAttention blocks.One of ``'chunk'``, ``'window'`` or ``None``. Default is ``'chunk'``
       :parama pe:Type of positional encoding to add.Must be one of ``'original'``, ``'regular'`` or ``None``. Default is ``None``.

       """
    def __init__(self, d_input: int, d_model: int, hidden_dim: int, d_output: int, q: int, v: int, h: int, M: int,
                 attention_size: int = None, dropout: float = 0.3, chunk_mode: bool = True, pe: str = 'original'):
        super().__init__()


        self._d_model = d_model
        self.layers_encoding = nn.ModuleList(
            [Encoder(d_model, q, v, h, attention_size=attention_size, dropout=dropout, chunk_mode=chunk_mode) for _ in
             range(M)])
        self._embedding = nn.Linear(d_input, d_model)
        self._linear = nn.Linear(d_model, d_output)

        pe_functions = {
            'original': generate_original_PE,
            'regular': generate_regular_PE,
        }
        if pe in pe_functions.keys():
            self._generate_PE = pe_functions[pe]
        elif pe is None:
            self._generate_PE = None
        else:
            raise NameError(
                f'PE "{pe}" not understood. Must be one of {", ".join(pe_functions.keys())} or None.')

        self.name = 'Truncated_Cauchy_self_attention_block'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        self-attention block
        :param x: class:`torch.Tensor` of shape (batch_size, H).
        :return: output: Output tensor with shape (batch_size, H, d_output).
        """
        x = torch.unsqueeze(x, 2)
        K = x.shape[1]  # 3 4 8

        # Input layer
        encoding = self._embedding(x)

        # Position encoding
        if self._generate_PE is not None:
            positional_encoding = self._generate_PE(K, self._d_model)
            positional_encoding = positional_encoding.to(encoding.device)
            encoding.add_(positional_encoding)

        # Encoder；layers
        for layer in self.layers_encoding:
            encoding = layer(encoding)
        output = self._linear(encoding)    #reshape the output of encoder；layer
        output = torch.sigmoid(output)
        return output