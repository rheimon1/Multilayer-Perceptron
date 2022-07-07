import numpy as np

class Mlp:
  """
  Classe que implementa a arquitetura de uma Mlp, implementando métodos que realizam
  as tarefas esperados por uma Rede Neural do tipo Multilayer Perceptron.
  """

  def __init__(self, layers, activation_function, file_provider, learning_rate=1e-2, threshold=1e-3):
    """
    Parameters
    ----------
    layers : Camadas que vão compor a arquitetura da Multilayer Perceptron
    activation_function : Classe com função de ativação e sua derivada
    file_provider : Instância da classe FileProvider
    learning_rate : Taxa de aprendizado. Valor Padrão = 1e-2
    threshold : Limite de erro. Valor Padrão = 1e-3
    """

    self.activation_function = activation_function.f
    self.d_activation_function = activation_function.df
    self.file_provider = file_provider
    self.learning_rate = learning_rate
    self.threshold = threshold
    self.init_layers(layers)    

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

    # Define as informações de pesos de entrada para a camada escondida
    self.hidden_layer_weights = np.random.random(input_length*hidden_length).reshape(input_length, hidden_length)
    self.hidden_layer_in = np.empty(hidden_length)
    self.hidden_layer_out = np.empty(hidden_length)
    self.hidden_layer_bias = np.random.random(hidden_length)

    # Define as informações de pesos para a camada de saída
    self.output_layer_weights = np.random.random(hidden_length*output_length).reshape(hidden_length, output_length)
    self.output_layer_in = np.empty(hidden_length)
    self.output_layer_bias = np.random.random(output_length)

    file_content_initial_weights = f"Pesos de entrada da Camada Escondida\n{self.hidden_layer_weights}\n\n"\
      f"Biases da Camada Escondida\n{self.hidden_layer_bias}\n\n\n"\
      f"Pesos de entrada da Camada de Saída\n{self.output_layer_weights}\n\n"\
      f"Biases da Camada de Saída\n{self.output_layer_bias}\n\n\n"
    self.file_provider.write_txt("docs/pesos-iniciais", file_content_initial_weights)

  def feedforward(self, input_signal):
    """Nesse método ocorre o processe de feedforward, em que ocorrerão os processos de soma de entradas
    ponderadas, aplicação de função de ativação para computar sinal de saída e envio para a próxima camada. 
    Isso ocorre para a camada escondida e camada de saída.
    
    Parameters
    ----------
    input_signal : Sinal de entrada (Xi) que é dissipado para todas as unidades da camada escondida"""

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
    # Com isso, tem-se a saída obtida para uma determinada entrada do conjunto de dados 
    self.output_layer_out = self.activation_function(self.output_layer_in)

    return self.output_layer_out

  def backpropagation(self, error, input_signal):
    """Este método vai otimizar os pesos para que a rede neural consiga mapear de forma correta as
    entradas para a saída. Para isso, vai ser calculado de ínicio o termo de informação do erro que
    leva em consideração a diferença entre a saída esperada e a saída obtida multiplicado pela derivada
    do Y_in(self.output_layer_in) que foi calculado no feedforward. Com o termo definido, calcula-se então a
    correção de pesos e bias que são usados para computar o sinal de entrada da camada de saída 
    (delta_output_layer_weights e delta_output_layer_bias) e encaminha esse termo para a camada anterior 
    (camada escondida) que vai também calcular a correção de pesos e bias que ela usa para computar o sinal de 
    entrada.

    Parameters
    ----------
    error : Diferença entre a saída esperada e a saída obtida pelo feedforward
    input_signal : Sinal de entrada (Xi) que é dissipado para todas as unidades da camada escondida"""
  
    # Termo de informação de erro (deltinha k)
    small_delta_k = error * self.d_activation_function(self.output_layer_in)

    # Cálculo da correção de pesos e bias para a camada de saída
    delta_output_layer_weights = self.learning_rate * np.matmul(
      self.hidden_layer_out.reshape(-1, 1), small_delta_k.reshape(1, -1)
    )
    delta_output_layer_bias = self.learning_rate * small_delta_k


    small_delta_in_j = np.matmul(self.output_layer_weights, small_delta_k)
    small_delta_j = small_delta_in_j * self.d_activation_function(self.hidden_layer_in)

    # Cálculo da correção de pesos e bias para a camada escondida
    delta_hidden_layer_weights = self.learning_rate * np.matmul(
      input_signal.reshape(-1, 1), small_delta_j.reshape(1, -1)
    )
    delta_hidden_layer_bias = self.learning_rate * small_delta_j


    # Atualização dos pesos e bias que são usados nas camada de saída e na escondida. Isso envolve
    # computar um novo valor que leva em conta o valor antigo e a correção calculada anteriormente.
    self.output_layer_weights = self.output_layer_weights + delta_output_layer_weights
    self.output_layer_bias = self.output_layer_bias + delta_output_layer_bias

    self.hidden_layer_weights = self.hidden_layer_weights + delta_hidden_layer_weights
    self.hidden_layer_bias = self.hidden_layer_bias + delta_hidden_layer_bias

  
  def train(self, inputs, labels, threshold=0.01):
    squaredError = 2 * threshold
    epochs = 0

    file_content_error = ""

    while(squaredError > threshold):
      squaredError = 0

      for i in range(len(inputs)):
        x = inputs[i]
        y_expected = labels[i]

        y_obtained = self.feedforward(x)

        error = np.subtract(y_expected, y_obtained)
        squaredError += np.sum(np.power(error, 2))
        
        self.backpropagation(error, x)

      squaredError = squaredError / len(inputs)

      print(f"Erro médio quadrado = {squaredError}")
      
      file_content_error += f"Erro médio quadrado = {squaredError}\n"

      epochs = epochs + 1

    file_content_error += f"\nTotal de iterações = {epochs}"

    self.file_provider.write_txt("docs/erro-cometido-por-iteracao", file_content_error)

    file_content_final_weights = f"Pesos de entrada da Camada Escondida\n{self.hidden_layer_weights}\n\n"\
      f"Biases da Camada Escondida\n{self.hidden_layer_bias}\n\n\n"\
      f"Pesos de entrada da Camada de Saída\n{self.output_layer_weights}\n\n"\
      f"Biases da Camada de Saída\n{self.output_layer_bias}\n\n\n"
    self.file_provider.write_txt("docs/pesos-finais", file_content_final_weights)

