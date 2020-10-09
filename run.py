# We will automate all the logistic regression baseline experiments for different event types and their subtasks
# For each Event type we will first run the data_preprocessing and 
# then run the logistic regression classifier for each subtask that has few (non-zero) positive examples
# We will save all the different classifier models, configs and results in separate directories
# Finally when all the codes have finished we will aggregate all the results and save the final metrics in csv file

from model.utils import make_dir_if_not_exists, load_from_pickle, load_from_json
import os
import csv
import subprocess
import argparse
import json

import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


parser = argparse.ArgumentParser()

parser.add_argument(
  "-c", "--config", help="Path to the config file that contains the experiment details", type=str,
  default='configs/hsw_iter2_t4.json'
)
# tested_positive,tested_negative,can_not_test,death,cure
parser.add_argument("-t", "--tasks", help="Tasks to run", type=str, default='cure')
args = parser.parse_args()

config = json.load(open(args.config))
team_name = 'HLTRI_SARCASM'
team_prediction_folder = os.path.join('data', team_name)
if not os.path.exists(team_prediction_folder):
  os.mkdir(team_prediction_folder)

task_type_to_datapath_dict = {
  "cure": {
    "data_in_file": "./data/cure_and_prevention-add_text-sarcasm.jsonl",
    "processed_out_file": "./data/cure_and_prevention-sarcasm.pkl",
    "predict_data_in_file": "./data/shared_task_test_set_final/shared_task-test-cure.jsonl",
    "predict_processed_out_file": "./data/shared_task_test_set_final/shared_task_cure.pkl",
    "predict_file": os.path.join(team_prediction_folder, f'{team_name}-cure-sarcasm.jsonl')
  }
}

# REDO_DATA_FLAG = True
REDO_DATA_FLAG = True
RETRAIN_FLAG = True
PREDICT_FLAG = False
# run_tasks = {"tested_positive", "tested_negative", "can_not_test", "death", "cure"}
run_tasks = set(args.tasks.split(','))
model_type = config['model_type']
run_name = config['run_name']
gpu_id = config['gpu_id']
pre_model_name = config['pre_model_name']
model_flags = config['model_flags']

# We will save all the tasks and subtask's results and model configs in this dictionary
all_task_results_and_model_configs = dict()
# We will save the list of question_tags AKA subtasks for each event AKA task in this dict
all_task_question_tags = dict()
for taskname, task_dict in task_type_to_datapath_dict.items():
  if taskname not in run_tasks:
    continue
  data_in_file = task_dict['data_in_file']
  processed_out_file = task_dict['processed_out_file']
  predict_data_in_file = task_dict['predict_data_in_file']
  predict_processed_out_file = task_dict['predict_processed_out_file']
  predict_file = task_dict['predict_file']

  if not os.path.exists(processed_out_file) or REDO_DATA_FLAG:
    data_preprocessing_cmd = f"python -m model.data_preprocessing -d {data_in_file} -s {processed_out_file}"
    logging.info(data_preprocessing_cmd)
    os.system(data_preprocessing_cmd)
  else:
    logging.info(f"Preprocessed data for task {taskname} already exists at {processed_out_file}")

  if PREDICT_FLAG and (not os.path.exists(predict_processed_out_file) or REDO_DATA_FLAG):
    data_preprocessing_cmd = f"python -m model.data_preprocessing -pd" \
      f" -d {predict_data_in_file}" \
      f" -s {predict_processed_out_file}" \
      f" -ts {processed_out_file}"
    logging.info(data_preprocessing_cmd)
    os.system(data_preprocessing_cmd)
  elif PREDICT_FLAG:
    logging.info(f"Preprocessed data for prediction task {taskname} already exists at {predict_processed_out_file}")

  # We will store the list of subtasks for which we train the classifier
  tested_tasks = list()
  logging.info(f"Running Mutlitask BERT Entity Classifier model on {processed_out_file}")
  # NOTE: After fixing the USER and URL tags
  output_dir = os.path.join("results", f"multitask_{model_type}_{run_name}_entity_classifier_fixed_sarcasm", taskname)
  make_dir_if_not_exists(output_dir)
  results_file = os.path.join(output_dir, "results.json")
  model_config_file = os.path.join(output_dir, "model_config.json")
  # Execute the Bert entity classifier train and test only if the results file doesn't exists
  # After fixing the USER and URL tags
  multitask_bert_cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python -m model.multitask_{model_type}_entity_classifier " \
    f"-d {processed_out_file} " \
    f"-t {taskname} " \
    f"-o {output_dir} " \
    f"-s saved_models/multitask_{model_type}_{run_name}_entity_classifier_fixed_sarcasm/{taskname} " \
    f"-pm {pre_model_name} " \
    f"-mf '{json.dumps(model_flags)}' "
  if RETRAIN_FLAG:
    multitask_bert_cmd += " -r"
  if PREDICT_FLAG:
    multitask_bert_cmd += f"-p -pd {predict_processed_out_file} -po {predict_file}"
  logging.info(f"Running: {multitask_bert_cmd}")
  try:
    retcode = subprocess.call(multitask_bert_cmd, shell=True)
  # os.system(multitask_bert_cmd)
  except KeyboardInterrupt:
    exit()





