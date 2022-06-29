import numpy as np

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