import torch
import torch.nn as nn
import numpy as np
from typing import Optional
import torch.nn.functional as F

beta = 1/5
def get_dist_mask_tile(sentence_len):
    """
        Calculate Cauchy matrix
        :param sentence_len: Length of attention matrix
        :return: dis_mask: Returns a matrix in which the elements obey the Cauchy distribution
        """
    row, col = torch.meshgrid(torch.arange(sentence_len), torch.arange(sentence_len))
    dis_mask = (row - col).abs()
    dis_mask =  1 /(1 + beta*np.power(dis_mask, 2))
    return dis_mask
# Laplace
# def get_dist_mask_tile(sentence_len):
#     row, col = torch.meshgrid(torch.arange(sentence_len), torch.arange(sentence_len))
#     dis_mask = (row - col).abs()
#     dis_mask = np.exp(-dis_alpha*dis_mask.float())
#     return dis_mask

class Self_attention_layer(nn.Module):
    """
    self-attention layer.
    Given 3 inputs of shape (batch_size, H, d_model), that will be used
    to compute query, keys and values, we output a self attention
    tensor of shape (batch_size, H, d_model).
    """
    def __init__(self,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int,
                 attention_size: int = None):
        """
        :param d_model: Dimension of the input vector.
        :param q: Dimension of all query matrix.
        :param v: Dimension of all value matrix.
        :param h: Number of heads.
        :param attention_size: Number of backward elements to apply attention.
        Deactivated if ``None``. Default is ``None``.
        """
        super().__init__()
        self._dopout = nn.Dropout(p=0.3)
        self._h = h

        # Query, keys and value matrices
        self._W_q = nn.Linear(d_model, q * self._h)
        self._W_k = nn.Linear(d_model, q * self._h)
        self._W_v = nn.Linear(d_model, v * self._h)

        # Output linear function
        self._W_o = nn.Linear(self._h * v, d_model)

        # Score placeholder
        self._scores = None

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[str] = None) -> torch.Tensor:
        """Propagate forward the input through the self-attention layer.
        We compute for each head the queries, keys and values matrices,
        followed by the Scaled Dot-Product. The result is concatenated
        and returned with shape (batch_size, H, d_model).

        :param query: Input tensor with shape (batch_size, H, d_model) used to compute queries.
        :param key: Input tensor with shape (batch_size, H d_model) used to compute keys.
        :param value: Input tensor with shape (batch_size, H, d_model) used to compute values.
        :param mask: Mask to apply on scores before computing attention. One of ``'subsequent'``, None. Default is None.
        :return: attention: Self attention tensor with shape (batch_size, H, d_model).
        """
        K = query.shape[1]

        # Compute Q, K and V, concatenate heads on batch dimension
        queries = torch.cat(self._W_q(query).chunk(self._h, dim=-1), dim=0)
        keys = torch.cat(self._W_k(key).chunk(self._h, dim=-1), dim=0)  #
        values = torch.cat(self._W_v(value).chunk(self._h, dim=-1), dim=0)

        # Add a new matrix which obeys Cauchy distribution to original attention matrix
        self._scores = torch.bmm(query, key.transpose(1, 2)) / np.sqrt(K)
        dist_mask_tile = get_dist_mask_tile(self._scores.shape[1])
        # diagonal_zero_mask = np.ones([self._scores.shape[1], self._scores.shape[1]]) -  np.diag([1.0] * self._scores.shape[1])
        # diagonal_zero_mask=torch.tensor(diagonal_zero_mask)
        # dist_mask_tile=dist_mask_tile*diagonal_zero_mask
        self._scores += dist_mask_tile.cuda()
        dir_mask = torch.triu(torch.ones((self._scores.shape[1], self._scores.shape[1])), diagonal=1).bool().cuda()
        self._scores = self._scores.masked_fill(dir_mask, float('-inf')).cuda()

        # Compute future mask
        if mask == "subsequent":
            future_mask = torch.triu(torch.ones((K, K)), diagonal=1).bool()
            future_mask = future_mask.to(self._scores.device)
            self._scores = self._scores.masked_fill(future_mask, float('-inf'))

        # Apply sotfmax
        self._scores = F.softmax(self._scores, dim=-1)
        attention = torch.bmm(self._scores, value)
        return attention

    @property
    def attention_map(self) -> torch.Tensor:
        """Attention map after a forward propagation,
        variable `score` in the original paper.
        """
        if self._scores is None:
            raise RuntimeError(
                "Evaluate the model once to generate attention map")
        return self._scores
class PositionwiseFeedForward(nn.Module):
    """A two-feed-forward-layer module """
    def __init__(self,
                 d_model: int,
                 d_ff: Optional[int] = 210):

        super().__init__()
        self._linear1 = nn.Linear(d_model, d_ff)
        self._linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._linear2(F.relu(self._linear1(x)))
class Encoder(nn.Module):
    """Encoder layer is made up of self-attn layer and feed forward and layerNorm (defined below)"""
    def __init__(self,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int,
                 attention_size: int = None,
                 dropout: float = 0.3,
                 chunk_mode: str = 'chunk'):
        super().__init__()
        self._selfAttention = Self_attention_layer(d_model, q, v, h, attention_size=attention_size)
        self._feedForward = PositionwiseFeedForward(d_model)

        self._layerNorm1 = nn.LayerNorm(d_model)
        self._layerNorm2 = nn.LayerNorm(d_model)

        self._dopout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Propagate the input through the Encoder layers.
        Apply the self-attention layer, add residual and normalize.
        Apply the Point-wise Feed Forward block, add residual and normalize.

        :param x: Input tensor with shape (batch_size, H, d_model).
        :return: x: Output tensor with shape (batch_size, H, d_model).
        """

        # each encoder layer
        residual = x
        x = self._selfAttention(query=x, key=x, value=x) # Cauchy self-attention layer
        x = self._dopout(x)
        x = self._layerNorm1(x + residual)

        # Feed forward
        residual = x
        x = self._feedForward(x)
        x = self._dopout(x)
        x = self._layerNorm2(x + residual)

        return x

    @property
    def attention_map(self) -> torch.Tensor:
        """Attention map after a forward propagation,
        variable `score` in the original paper.
        """
        return self._selfAttention.attention_map