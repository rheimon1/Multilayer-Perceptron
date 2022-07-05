from activation_function import SigmoidActivation
from csv_provider import CsvProvider
from mlp import Mlp
import numpy as np

class Main:
  """
  Classe que vai adaptar e tratar os dados para posteriormente ser encaminhados para a classe `Mlp`.
  Essa classe é muito importante para o uso da classe `Mlp`, pois ela organiza e encaminha as informações 
  no formato que a classe espera.
  """

  def __init__(
    self, 
    csv_provider: CsvProvider, 
    activation_function, 
    input_length, 
    output_length, 
    hidden_length,
    learning_rate,
    threshold
  ):
    """
    Parameters
    ----------
    csv_provider : Instãncia de classe que possibilita manipular arquivos csv
    activation_function : Instãncia de classe que implementa função de ativação, bem como sua derivada
    input_length : Define quantos elementos terão na camada de entrada
    output_length : Define quantos elementos terão na camada de saída
    hidden_length : Define quantos elementos terão na camada escondida
    learning_rate : Taxa de aprendizado
    threshold : Limite de erro
    """

    self.csv_provider = csv_provider

    self.input_length = input_length
    self.output_length = output_length
    self.hidden_length = hidden_length

    self.mlp = Mlp(
      layers={
        'input_length': input_length,
        'output_length': output_length,
        'hidden_length': hidden_length
      },
      activation_function=activation_function,
      learning_rate=learning_rate,
      threshold=threshold
    )

  def convert_str_values_to_int_in_matrix(self, matrix):
    """Converte os valores de uma matriz do tipo `String` para `Integer`. 

    Parameters
    ----------
    matrix : Matriz com os valores a ser convertidos
    """

    return [[int(i) for i in j] for j in matrix]

  def convert_negative_values_to_zero_in_matrix(self, matrix):
    """Converte os valores de uma matriz que são negativos (-1) para 0.

    Parameters
    ----------
    matrix : Matriz com os valores a ser convertidos
    """

    for i in range(len(matrix)):
      for j in range(len(matrix[i])):
        if (matrix[i][j] < 0):
          matrix[i][j] = 0

  def prepare_data(self, dataset):
    """Função responsável por converter os dados de um dataset para o formato esperado pela Classe `Mlp`.
    Isso envolve separar dados de entrada de dados que são os rótulos. Converter o tipo dos dados de
    String para Integer a fim de tornar possível os cálculos. Converter os valores negativos para zero e
    por fim definir o formato das matrizes com auxílio de uma biblioteca chamada `numpy` que vai facilitar
    as operações com matrizes e arrays.

    Parameters
    ----------
    dataset : Dados obtidos de alguma fonte externa que serão analisados, tratados e organizados. 
    """

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

  def train_data(self, path):
    """
    Função que vai ler os dados de um determinado caminho, obter um dataset disso, tratar esse dataset, organizando
    os dados no formato esperado pela classe `Mlp` e vai treinar esses dados.

    Parameters
    ----------
    path : Caminho que contém o arquivo com os dados que serão treinados
    """

    dataset = self.csv_provider.load(path)
    input_data, label_data = self.prepare_data(dataset)
    self.mlp.train(input_data, label_data)

  def predict(self, path):
    pass

csvProvider = CsvProvider()
sigmoidActivation = SigmoidActivation()
main = Main(csvProvider, sigmoidActivation,  2, 1, 2, 0.1, 1e-3)
main.train_data('xor.csv')