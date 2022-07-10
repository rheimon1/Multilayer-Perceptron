from activation_function import SigmoidActivation
from file_provider import FileProvider
from mlp import Mlp
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

class Main:
    """
    Classe que vai adaptar e tratar os dados para posteriormente ser encaminhados para a classe `Mlp`.
    Essa classe é muito importante para o uso da classe `Mlp`, pois ela organiza e encaminha as informações 
    no formato que a classe espera.
    """

    def __init__(
        self,
        input_length, 
        hidden_length,
        output_length, 
        learning_rate,
        threshold,
        file_provider: FileProvider, 
        activation_function,
    ):
        """
        Parameters
        ----------
        file_provider : Instãncia de classe que possibilita manipular arquivos
        activation_function : Instãncia de classe que implementa função de ativação, bem como sua derivada
        input_length : Define quantos elementos terão na camada de entrada
        output_length : Define quantos elementos terão na camada de saída
        hidden_length : Define quantos elementos terão na camada escondida
        learning_rate : Taxa de aprendizado
        threshold : Limite de erro
        """

        self.file_provider = file_provider

        self.input_length = input_length
        self.output_length = output_length
        self.hidden_length = hidden_length

        file_content = f"Tamanho das camadas\n\tEntrada = {input_length}\n\tEscondida = {hidden_length}\n"\
            f"\tSaída = {output_length}\n\nTaxa de aprendizado = {learning_rate}\n\nLimiar do erro = {threshold}"\
            f"\n\nFunção de ativação = {activation_function.get_name()}"
        self.file_provider.write_txt("docs/parametros", file_content)

        self.mlp = Mlp(
            layers={
                'input_length': input_length,
                'output_length': output_length,
                'hidden_length': hidden_length
            },
            activation_function=activation_function,
            file_provider=file_provider,
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

        inputs = [ [] for _ in range(len(dataset)) ]
        labels = [ [] for _ in range(len(dataset)) ]

        for i in range(len(dataset)):
            inputs[i] = dataset[i][0:-self.output_length]
            labels[i] = dataset[i][-self.output_length:]

        inputs = self.convert_str_values_to_int_in_matrix(inputs)
        labels = self.convert_str_values_to_int_in_matrix(labels)

        self.convert_negative_values_to_zero_in_matrix(inputs)

        return inputs, labels

    def train_data(self, path, num_of_lines):
        """Função que vai ler os dados de um determinado caminho, obter um dataset disso, tratar esse dataset, organizando
        os dados no formato esperado pela classe `Mlp` e vai treinar esses dados.

        Parameters
        ----------
        path : Caminho que contém o arquivo com os dados que serão treinados
        num_of_lines : número de linhas para ser lidas do arquivo"""

        dataset = self.file_provider.load_csv(path)
        inputs, labels = self.prepare_data(dataset)
        inputs_to_train = []

        for i in range(0, num_of_lines):
            inputs_to_train.append(inputs[i])

        # Converte list para formato numpy.ndarray a fim de facilitar operaçoes matemáticas sobre essas estruturas
        inputs_to_train = np.array(inputs_to_train)
        labels = np.array(labels)

        self.mlp.train(inputs_to_train, labels)

    def test_data(self, path):
        """Função que vai ler os dados de um determinado caminho, obter um dataset disso, tratar esse dataset, organizando
        os dados no formato esperado pela classe `Mlp` e vai testar esses dados.

        Parameters
        ----------
        path : Caminho que contém o arquivo com os dados que serão treinados"""

        dataset = self.file_provider.load_csv(path)
        inputs, labels = self.prepare_data(dataset)
        labels = np.array(labels)
        result = self.mlp.feedforward(inputs)
        self.content_file_test_results = ""
        self.content_file_test_results += f"Testes para arquivo - {path}\n\n"
        file_content_result = self.format_test_results_output(labels, result) + "\n\n"
        self.content_file_test_results += file_content_result
        self.content_file_test_results += f"Matriz de confusão\n{str(confusion_matrix(labels.argmax(axis=1), result.argmax(axis=1)))}\n"
        self.file_provider.write_txt('docs/resultados-'+path, self.content_file_test_results)
        
    def format_test_results_output(self, labels, results):

        # Arredonda os resultados da saída do método feedforward do mlp para o inteiro mais próximo
        label_values = ['A', 'B', 'C', 'D', 'E', 'J', 'K']
        results = np.round_(results)

        expected_results = []
        for row in labels:
            for index in range(len(row)):
                if row[index] == 1:
                    expected_results.append(index)

        obtained_results = []
        for row in results:
            temp = []
            for index in range(len(row)):
                if row[index] == 1:
                    temp.append(index)
            obtained_results.append(temp)
        
        result_content_file = ""
        for i in range(len(expected_results)):
            expected_result = expected_results[i]
            expected_result_value = label_values[expected_result]

            result_text = f"Label: {expected_result_value} - Resultado obtido: "
            obtained_values_text = ""
            for j in range(len(obtained_results[i])):
                obtained_result = obtained_results[i][j]
                obtained_result_value = label_values[obtained_result]
                if (j > 0):
                    obtained_values_text += ", "
                obtained_values_text += f"{obtained_result_value}"
            if obtained_values_text == "":
                obtained_values_text = "Nenhum"
            result_content_file += result_text + obtained_values_text + "\n"

        return result_content_file


file_provider = FileProvider()
sigmoid_activation = SigmoidActivation()
main = Main(63, 15, 7, 0.1, 0.05, file_provider, sigmoid_activation)
main.train_data('problemXOR')
main.train_data('caracteres-limpo', 14)
main.test_data('caracteres-limpo')
main.test_data('caracteres-ruido')
main.test_data('caracteres_ruido20')


