
### a standalone script for evaluation

import argparse
import json


def read_json_line(path):
    output = []
    with open(path, 'r') as f:
        for line in f:
            output.append(json.loads(line))
    return output


parser = argparse.ArgumentParser()
parser.add_argument("-l", "--labels", help="Path to the dataset labels", type=str, required=True)
parser.add_argument("-g", "--golden", help="Path to the golden labels", type=str, required=True)
args = parser.parse_args()


def main():
    ## read in files
    labels = read_json_line(args.labels)

    with open(args.golden, 'w') as f:
      for label in labels:
        text = label['text']
        ann = label['annotations']
        golden_annotation = {}
        for st_name, st_vals in ann.items():
          gold_vals = []
          for val in st_vals:
            if isinstance(val, list):
              s, e = val
              val = text[s:e]
            gold_vals.append(val)
          golden_annotation[st_name] = gold_vals
        ex = {
          'id': label['id'],
          'golden_annotation': golden_annotation
        }
        f.write(json.dumps(ex) + '\n')


if __name__ == '__main__':
    main()
