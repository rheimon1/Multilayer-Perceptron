import numpy as np
from helper import Helper

class Mlp:
  def __init__(self, input_length, hidden_length, output_length, activation_function, d_activation_function):
    self.hidden_layer_weights = np.random.random(input_length*hidden_length).reshape(input_length, hidden_length)
    self.hidden_layer_in = np.empty(hidden_length)
    self.hidden_layer_out = np.empty(hidden_length)
    self.hidden_layer_bias = np.random.random(hidden_length)

    self.output_layer_weights = np.random.random(hidden_length*output_length).reshape(hidden_length, output_length)
    self.output_layer_in = np.empty(hidden_length)
    self.output_layer_bias = np.random.random(output_length)
    
    self.activation_function = activation_function
    self.d_activation_function = d_activation_function

  def feed_forward(self, Xp):
    # Hidden Layer
    self.hidden_layer_in = np.matmul(Xp, self.hidden_layer_weights) + self.hidden_layer_bias
    self.hidden_layer_out = self.activation_function(self.hidden_layer_in)

    # Output Layer
    self.output_layer_in = np.matmul(self.hidden_layer_out, self.output_layer_weights) + self.output_layer_bias
    output_layer_out = self.activation_function(self.output_layer_in)

    return output_layer_out

  def back_propagation(self, error, data, eta=0.1):
    delta_k = error * self.d_activation_function(self.output_layer_in)

    delta_output_layer_bias = eta * delta_k
    delta_output_layer_weights = eta * np.matmul(self.hidden_layer_out.reshape(-1, 1), delta_k.reshape(1, -1))

    delta_in_j = np.matmul(self.output_layer_weights, delta_k)
    delta_j = delta_in_j * self.d_activation_function(self.hidden_layer_in)

    delta_hidden_layer_bias = eta * delta_j
    delta_hidden_layer_weights = eta * np.matmul(data.reshape(-1, 1), delta_j.reshape(1, -1))

    self.output_layer_weights = self.output_layer_weights + delta_output_layer_weights
    self.output_layer_bias = self.output_layer_bias + delta_output_layer_bias

    self.hidden_layer_weights = self.hidden_layer_weights + delta_hidden_layer_weights
    self.hidden_layer_bias = self.hidden_layer_bias + delta_hidden_layer_bias

  
  def fit(self, inputs, labels, threshold=0.1):
    squaredError = 2 * threshold
    counter = 0

    while(squaredError > threshold):
      squaredError = 0

      for i in range(len(inputs)):
        Xp = inputs[i]
        Yp = labels[i]

        results = self.feed_forward(Xp)
        Op = results

        error = np.subtract(Yp, Op)
        squaredError = squaredError + np.sum(np.power(error, 2))
        
        self.back_propagation(error, Xp)

      squaredError = squaredError / len(inputs)

      print(f"Erro médio quadrado = {squaredError}")

      counter = counter + 1

output_length = 1

helper = Helper()
dataset = helper.load_csv('xor.csv')
inputs = [ [] for _ in range(len(dataset)) ]
labels = [ [] for _ in range(len(dataset)) ]

for i in range(len(dataset)):
  inputs[i] = dataset[i][0:-output_length] 
  labels[i] = dataset[i][-output_length:]

inputs = [ [ int(inputs[i][j]) for j in range(len(inputs[0])) ] for i in range(len(inputs)) ]
labels = [ [ int(labels[i][j]) for j in range(len(labels[0])) ] for i in range(len(labels)) ]

inputs = helper.convert_negative_to_zero(inputs)

inputs = np.array(inputs)
labels = np.array(labels)

mlp = Mlp(2, 2, 1, helper.sigmoid, helper.sigmoid_prime)

# Xp = inputs[0]
# print(f'Xp {Xp}')
# hidden = mlp.layers['hidden']
# print(f'hidden: {hidden}')
# print(Xp @ hidden)

print(f'inputs: {inputs}')
print(f'labels: {labels}')

mlp.fit(inputs=inputs, labels=labels, threshold=0.001)
print(f"Pesos da camada escondida = {mlp.hidden_layer_weights}")
print(f"Pesos da camada de saída = {mlp.output_layer_weights}")


