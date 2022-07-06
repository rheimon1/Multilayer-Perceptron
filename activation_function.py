import numpy as np

"""A utilização da biblioteca numpy vai facilitar as operações envolvendo vetores. Assim, nas funções
abaixo quando se chega um parâmetro na forma de vetor, o método já vai aplicar a operação para todos
os elementos do vetor.
"""

def f_sigmoid(x):
  return 1 / (1 + np.exp(-x))

def df_sigmoid(x):
  return f_sigmoid(x) * (1 - f_sigmoid(x))

class SigmoidActivation:
  @staticmethod
  def f(x):
    return f_sigmoid(x)

  @staticmethod
  def df(x):
    return df_sigmoid(x)

  @staticmethod
  def get_name():
    return "Sigmoid"