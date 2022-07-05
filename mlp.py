import numpy as np

class Mlp:
  """
  Classe que implementa a arquitetura de uma Mlp, implementando métodos que realizam
  as tarefas esperados por uma Rede Neural do tipo Multilayer Perceptron.
  """

  def __init__(self, layers, activation_function, learning_rate=0.1, threshold=1e-3):
    """
    Parameters
    ----------
    layers : Camadas que vão compor a arquitetura da Multilayer Perceptron
    activation_function : Classe com função de ativação e sua derivada
    learning_rate : Taxa de aprendizado. Valor Padrão = 0.01
    threshold : Limite de erro. Valor Padrão = 1e-3
    """

    self.init_layers(layers)    
    self.activation_function = activation_function.f
    self.d_activation_function = activation_function.df
    self.learning_rate = learning_rate
    self.threshold = threshold

  def init_layers(self, layers):
    """Configura as camadas de entrada, saída e escondida inicialmente. Isso envolve atribuir pesos
    para as camadas de saída e escondida com valores aleatórios e definir as dimensões das matrizes
    que vão compor esses pesos de acordo com os tamanhos pre-definidos de cada camada. Além disso, 
    aqui também é definido a estrutura dos biases para as duas camadas acima, bem como são atribuídos 
    valores aleatórios para eles.
    
    Parameters
    ----------
    layers : Camadas que vão compor a arquitetura da Multilayer Perceptron
    """

    input_length = layers['input_length']
    output_length = layers['output_length']
    hidden_length = layers['hidden_length']

    # Define as informações para a camada escondida
    self.hidden_layer_weights = np.random.random(input_length*hidden_length).reshape(input_length, hidden_length)
    self.hidden_layer_in = np.empty(hidden_length)
    self.hidden_layer_out = np.empty(hidden_length)
    self.hidden_layer_bias = np.random.random(hidden_length)

    # Define as informações para a camada de saída
    self.output_layer_weights = np.random.random(hidden_length*output_length).reshape(hidden_length, output_length)
    self.output_layer_in = np.empty(hidden_length)
    self.output_layer_bias = np.random.random(output_length)

  def feed_forward(self, input_signal):
    """Nesse método ocorre o processe de feedforward, em que ocorrerão os processos de soma de entradas ponderadas, aplicação de função de ativação para computar sinal de saída e envio para a próxima cada. Isso ocorre para a camada escondida e camada de saída.
    
    Parameters
    ----------
    input_signal : Sinal de entrada (Xi) que é dissipado para todas as unidades da camada escondida
    """

    # Para cada unidade da camada escondida(Z), soma as entradas ponderadas junto ao bias. Isso
    # corresponde a uma multiplicação de matrizes dos sinais de entrada (Xi) pelos pesos definidos
    # da camada escondida (Vij) e o resultado disso é somado ao bias (V0j) que resultará no Z_in
    self.hidden_layer_in = np.matmul(input_signal, self.hidden_layer_weights) + self.hidden_layer_bias
    
    # Com o Z_in calculado no passo anterior, aplica-se a função de ativação para computador o sinal
    # de saída da camada escondida (Zj). Esse sinal será encaminhado para a camada de saída para seguir
    # com o processo.
    self.hidden_layer_out = self.activation_function(self.hidden_layer_in)



    # Para cada unidade da camada de saída (Y), soma as entradas ponderadas junto ao bias. Isso
    # corresponde também a uma multiplicação de matrizes dos sinais de entradas vindo da camada
    # escondida (Zj) pelos pesos definidos da camada de saída (Wjk) e o resultado disso é somado
    # ao bias (W0k) que resultará no Y_in
    self.output_layer_in = np.matmul(self.hidden_layer_out, self.output_layer_weights) + self.output_layer_bias
    
    # Com o Y_in calculado anteriormente, aplica-se a função de ativação para computar o sinal de saída.
    # Essa será a saída final
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


