import argparse
from collections import OrderedDict

import os
import json

parser = argparse.ArgumentParser()

parser.add_argument("-m", "--model_name", type=str, required=True)
parser.add_argument("-r", "--results_path", type=str, default='results')
parser.add_argument("-mt", "--model_type", type=str, default='bert')
parser.add_argument("-t", "--task_name", type=str, default='tested_positive')
args = parser.parse_args()


results_path = args.results_path
model_name = args.model_name
model_type = args.model_type
task_name = args.task_name
input_path = os.path.join(results_path, f'multitask_{model_type}_{model_name}_entity_classifier_fixed', task_name, 'results.json')
rs = json.load(open(input_path))

non_sub_tasks = {'best_dev_threshold', 'best_dev_F1s', 'dev_t_F1_P_Rs'}
stats = ['P', 'R', 'F1']
cs = {'TP': 0, 'FP': 0, 'FN': 0}
headers = []
rows = []
for name, sub_task in rs.items():
  if name in non_sub_tasks:
    continue
  print(name)
  for stat in stats:
    headers.append(f'{name}-{stat}')
    value = sub_task[stat]
    rows.append(f'{value:.4f}')
    print(f'{stat}: {value:.4f}')
  for count_stat in cs:
    cs[count_stat] += sub_task[count_stat]
  print()

for c, v in cs.items():
  print(f'{c}: {v:.0f}')

front_headers = [task_name]
front_rows = [model_name]
micro_p = cs['TP'] / (cs['TP']+cs['FP'])
print()
print(f'Micro-P: {micro_p:.4f}')
front_headers.append('Micro-P')
front_rows.append(f'{micro_p:.4f}')
micro_r = cs['TP'] / (cs['TP']+cs['FN'])
print(f'Micro-R: {micro_r:.4f}')
front_headers.append('Micro-R')
front_rows.append(f'{micro_r:.4f}')
micro_f1 = (2.0 * (micro_p * micro_r)) / (micro_p + micro_r)
print(f'Micro-F1: {micro_f1:.4f}')
front_headers.append('Micro-F1')
front_rows.append(f'{micro_f1:.4f}')
headers = front_headers + headers
rows = front_rows + rows
print()
print('\t'.join(headers))
print('\t'.join(rows))