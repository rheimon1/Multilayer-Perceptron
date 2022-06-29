from csv import reader

class CsvProvider:
  def load(self, filename):
    dataset = list()
    with open(filename, encoding='utf-8-sig') as file:
      csv_reader = reader(file)
      for row in csv_reader:
        dataset.append(row)
      return dataset