def read_data_file(datafile):
    splitlines = datafile.read().splitlines()
    sentences = [line.decode('utf-8', 'ignore').lower() for line in splitlines]
    return sentences
