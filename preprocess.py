import argparse
import csv

from tqdm import tqdm

from preprocessing import BasicHtmlPreprocessor

parser = argparse.ArgumentParser()
parser.add_argument('--preprocessor_name', default='BasicHtmlPreprocessor')
parser.add_argument('-i')
parser.add_argument('--train_output', required=True)
parser.add_argument('--test_output')
parser.add_argument('--test_size', default=0)

args = parser.parse_args()

prepr_name: str = args.preprocessor_name
input_file: str = args.i
train_output_file: str = args.train_output
test_size: int = args.test_size

if prepr_name == 'BasicHtmlPreprocessor':
    preprocessor = BasicHtmlPreprocessor()
else:
    raise NotImplementedError('This preprocessor is not implemented')


with open(train_output_file, 'w') as train_out,  open(input_file, 'r') as inp:
    reader = csv.reader(inp)
    train_writer = csv.writer(train_out)

    header = next(reader)

    if test_size:
        test_output_file: str = args.test_output
        test_out = open(test_output_file, 'w')
        test_writer = csv.writer(test_out)
        test_writer.writerow(header)

        for i, (text, title) in tqdm(enumerate(reader)):
            test_writer.writerow([preprocessor.transform(text), title.lower()])
            if i == test_size:
                break

    train_writer.writerow(header)

    for text, title in tqdm(reader):
        train_writer.writerow([preprocessor.transform(text), title.lower()])


