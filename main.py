from helper import Helper
from mlp import Mlp

class Main:
  def __init__(self, helper:Helper, input_length, output_length, hidden_length, activation_function, d_activation_function):
    self.helper = helper
    self.mlp = Mlp(
      input_length=input_length, 
      output_length=output_length, 
      hidden_length=hidden_length,
      activation_function=helper.sigmoid,
      d_activation_function=helper.sigmoid_prime
    )

  def prepare_data(dataset):
    
    
  def execute(self, filename):
    dataset = self.helper.load_csv(filename)

    training_input = 

  