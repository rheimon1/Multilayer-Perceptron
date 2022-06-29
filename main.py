from activation_function import SigmoidActivation
from csv_provider import CsvProvider
from mlp import Mlp
import numpy as np

class Main:
  def __init__(
    self, 
    csv_provider: CsvProvider, 
    activation_function: SigmoidActivation, 
    input_length, 
    output_length, 
    hidden_length,
    learning_rate,
    threshold
  ):
    self.csv_provider = csv_provider

    self.input_length = input_length
    self.output_length = output_length
    self.hidden_length = hidden_length

    layers = {
      'input_length': input_length,
      'output_length': output_length,
      'hidden_length': hidden_length
    }

    self.mlp = Mlp(
      layers=layers,
      activation_function=activation_function,
      learning_rate=learning_rate,
      threshold=threshold
    )

  def convert_str_values_to_int_in_matrix(self, matrix):
    return [[int(i) for i in j] for j in matrix]

  def convert_negative_values_to_zero_in_matrix(self, matrix):
    for i in range(len(matrix)):
      for j in range(len(matrix[i])):
        if (matrix[i][j] < 0):
          matrix[i][j] = 0

  def prepare_data(self, dataset):
    input_data = [ [] for _ in range(len(dataset)) ]
    label_data = [ [] for _ in range(len(dataset)) ]

    for i in range(len(dataset)):
      input_data[i] = dataset[i][0:-self.output_length]
      label_data[i] = dataset[i][-self.output_length]

    input_data = self.convert_str_values_to_int_in_matrix(input_data)
    label_data = self.convert_str_values_to_int_in_matrix(label_data)

    self.convert_negative_values_to_zero_in_matrix(input_data)

    # Converte list para formato numpy.ndarray a fim de facilitar operaçoes matemáticas sobre essas estruturas
    input_data = np.asarray(input_data)
    label_data = np.asarray(label_data)

    return input_data, label_data

  def train_data(self, filename):
    dataset = self.csv_provider.load(filename)
    input_data, label_data = self.prepare_data(dataset)
    self.mlp.train(input_data, label_data)

csvProvider = CsvProvider()
sigmoidActivation = SigmoidActivation()
main = Main(csvProvider, sigmoidActivation,  2, 1, 2, 0.1, 1e-3)
main.train_data('xor.csv')