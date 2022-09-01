__author__ = "Tanay Rastogi"
"""
The puprose of the file is to create a ANN model in pytorch.

REFERENCE
 - https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
 - https://machinelearningmastery.com/pytorch-tutorial-develop-deep-learning-models/
"""

import torch
from torch import nn
torch.manual_seed(123)

class ANN(nn.Module):
  '''
    Simple Neural Network with 1 Hidden Layer
    
    INPUTS:
        input_dim      (int)   : Number of featrues in input.
        output_dim     (int)   : Number of outputs labels.
        hidden_dims    (list)  : List of number of nodes in each hidden layer. 
  '''
  def __init__(self, input_dim:int, output_dim:int, hidden_dims:list):
    super().__init__()

    ## Save variables ##
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.hidden_dims = hidden_dims

    ## Model ##
    current_dim = self.input_dim
    self.layers = nn.ModuleList()
    for hdim in self.hidden_dims:
      # Fully connected layer
      temp_layer = nn.Linear(current_dim, hdim)
      nn.init.kaiming_uniform_(temp_layer.weight, mode='fan_out', nonlinearity='relu')
      self.layers.append(temp_layer)
      # Use BatchNormalization
      temp_layer = nn.BatchNorm1d(hdim)
      self.layers.append(temp_layer)
      # Update current_dim
      current_dim = hdim

    temp_layer = nn.Linear(current_dim, output_dim)
    nn.init.xavier_uniform_(temp_layer.weight)
    self.layers.append(temp_layer)


  def forward(self, x):
    '''Forward pass'''
    # Input/Hidden layers
    for lyr in self.layers[:-1]:
      x = nn.functional.relu(lyr(x))
    x = nn.functional.softmax(self.layers[-1](x), dim=0)
    return x




if __name__ == "__main__":
  ## Testing model ###
  in_dims = 14
  out_dims = 7
  hidden = [20]
  model = ANN(input_dim=in_dims, output_dim=out_dims, hidden_dims=hidden)
  print(model)

  X = 20*torch.rand((5, in_dims))
  print(X)
  print(model(X))