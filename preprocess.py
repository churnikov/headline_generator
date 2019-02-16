import argparse
import csv

from tqdm import tqdm

from preprocessing import BasicHtmlPreprocessor

parser = argparse.ArgumentParser()
parser.add_argument('--preprocessor_name', default='BasicHtmlPreprocessor')
parser.add_argument('-i')
parser.add_argument('-o')

args = parser.parse_args()

prepr_name: str = args.preprocessor_name
input_file: str = args.i
output_file: str = args.o

if prepr_name == 'BasicHtmlPreprocessor':
    preprocessor = BasicHtmlPreprocessor()
else:
    raise NotImplementedError('This preprocessor is not implemented')


with open(output_file, 'w') as out, open(input_file, 'r') as inp:
    reader = csv.reader(inp)
    writer = csv.writer(out)

    writer.writerow(next(reader))

    for text, title in tqdm(reader):
        writer.writerow([preprocessor.transform(text), title.lower()])


