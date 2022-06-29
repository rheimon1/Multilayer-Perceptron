import numpy as np

class Mlp:
  def __init__(self, layers, activation_function, learning_rate=0.1, threshold=1e-3):
    self.init_layers(layers)    
    self.activation_function = activation_function.f
    self.d_activation_function = activation_function.df
    self.learning_rate = learning_rate
    self.threshold = threshold

  def init_layers(self, layers):
    input_length = layers['input_length']
    output_length = layers['output_length']
    hidden_length = layers['hidden_length']

    self.hidden_layer_weights = np.random.random(input_length*hidden_length).reshape(input_length, hidden_length)
    self.hidden_layer_in = np.empty(hidden_length)
    self.hidden_layer_out = np.empty(hidden_length)
    self.hidden_layer_bias = np.random.random(hidden_length)

    self.output_layer_weights = np.random.random(hidden_length*output_length).reshape(hidden_length, output_length)
    self.output_layer_in = np.empty(hidden_length)
    self.output_layer_bias = np.random.random(output_length)

  def feed_forward(self, data):
    # Hidden Layer
    self.hidden_layer_in = np.matmul(data, self.hidden_layer_weights) + self.hidden_layer_bias
    self.hidden_layer_out = self.activation_function(self.hidden_layer_in)

    # Output Layer
    self.output_layer_in = np.matmul(self.hidden_layer_out, self.output_layer_weights) + self.output_layer_bias
    self.output_layer_out = self.activation_function(self.output_layer_in)

    return self.output_layer_out

  def back_propagation(self, error, data):
    delta_k = error * self.d_activation_function(self.output_layer_in)

    delta_output_layer_bias = self.learning_rate * delta_k
    delta_output_layer_weights = self.learning_rate * np.matmul(self.hidden_layer_out.reshape(-1, 1), delta_k.reshape(1, -1))

    delta_in_j = np.matmul(self.output_layer_weights, delta_k)
    delta_j = delta_in_j * self.d_activation_function(self.hidden_layer_in)

    delta_hidden_layer_bias = self.learning_rate * delta_j
    delta_hidden_layer_weights = self.learning_rate * np.matmul(data.reshape(-1, 1), delta_j.reshape(1, -1))

    self.output_layer_weights = self.output_layer_weights + delta_output_layer_weights
    self.output_layer_bias = self.output_layer_bias + delta_output_layer_bias

    self.hidden_layer_weights = self.hidden_layer_weights + delta_hidden_layer_weights
    self.hidden_layer_bias = self.hidden_layer_bias + delta_hidden_layer_bias

  
  def train(self, inputs, labels, threshold=0.1):
    squaredError = 2 * threshold
    counter = 0

    while(squaredError > threshold):
      squaredError = 0

      for i in range(len(inputs)):
        x = inputs[i]
        y_expected = labels[i]

        y_obtained = self.feed_forward(x)

        error = np.subtract(y_expected, y_obtained)
        squaredError += np.sum(np.power(error, 2))
        
        self.back_propagation(error, x)

      squaredError = squaredError / len(inputs)

      print(f"Erro médio quadrado = {squaredError}")

      counter = counter + 1


