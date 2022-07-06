from csv import reader

class FileProvider:
  def load_csv(self, filename):
    dataset = list()
    with open(filename+'.csv', encoding='utf-8-sig') as file:
      csv_reader = reader(file)
      for row in csv_reader:
        dataset.append(row)
      return dataset
  
  def write_txt(self, filename, data):
    file = open(filename+'.txt', '+w')
    file.write(data)
    file.close()