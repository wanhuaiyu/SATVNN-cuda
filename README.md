# SATVNN

PyTorch implementation of SATVNN described in the paper entitled 
"Self-Attention based Time-Variant Neural Networks for Multi-Step Time Series Forecasting" 
by Changxia Gao, Ning Zhang, Youru Li, Feng Bian and Huaiyu Wan.


The key benifits of SATVNN are:
1. A novel Time-Variant structure is constructed to better learn the dynamics at different scales that span multiple time 
steps and to mitigate vanishing gradient problem.
2. A novel Self-Attention block with three different distributions is designed to reflect recent changes in temporal data, 
among which a novel Cauchy self-attention mechanism achieves better results than the Gaussian self-attention and the Laplace 
self-attention.
3. It is shown to out-perform state of the art deep learning models and statistical models.

## Files


- SATVNN.py: Contains the main class for SATVNN (Self-Attention based Time-Variant Neural Networks).
- calculateError.py: Contains helper functions to compute error metrics
- dataHelpers.py: Functions to generate the dataset use in demo.py and for for formatting data.
- demo.py: Trains and evaluates SATVNN on Water_usage dataset.
- Encoder.py: Encoder is made up of self-attention layer (Gaussian or Laplace or Cauchy), feed forward and Add & Norm.
- evaluate.py: Contains a rudimentary training function to train SATVNN.
- mse_loss.py: Calculates the mean squared error loss
- optimizer.py: Implements Open AI version of Adam algorithm with weight decay fix.
- train.py: Contains a rudimentary training function to train SATVNN.
- Self_attention_block.py: Contains self-attention block described in paper.


## Usage

Run the demo.py script to train and evaluate SATVNN model on Water_usage dataset. 

## Requirements

- Python 3.6
- Torch version 1.2.0
- NumPy 1.14.6.

