from csv import reader
import numpy as np

class Helper:
  def load_csv(self, filename):
    dataset = list()
    with open(filename, encoding='utf-8-sig') as file:
      csv_reader = reader(file)
      for row in csv_reader:
        dataset.append(row)
      return dataset

  def convert_negative_to_zero(self, data):
    for row in range(len(data)):
      for column in range(len(data[0])):
        if data[row][column] == -1 :
          data[row][column] = 0

        # Retorno do vetor convertido
    return data

  def sigmoid(self, x):
    return 1 / (1 + np.exp(-x))

  def sigmoid_prime(self, x):
    return self.sigmoid(x) * (1 - self.sigmoid(x))