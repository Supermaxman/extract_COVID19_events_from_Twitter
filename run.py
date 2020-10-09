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
    "predict_processed_out_file": "./data/shared_task_test_set_final/shared_task_cure-sarcasm.pkl",
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
  output_dir = os.path.join("results", f"multitask_{model_type}_{run_name}_entity_classifier_fixed", taskname)
  make_dir_if_not_exists(output_dir)
  results_file = os.path.join(output_dir, "results.json")
  model_config_file = os.path.join(output_dir, "model_config.json")
  # Execute the Bert entity classifier train and test only if the results file doesn't exists
  # After fixing the USER and URL tags
  multitask_bert_cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python -m model.multitask_{model_type}_entity_classifier " \
    f"-d {processed_out_file} " \
    f"-t {taskname} " \
    f"-o {output_dir} " \
    f"-s saved_models/multitask_{model_type}_{run_name}_entity_classifier_fixed/{taskname} " \
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
  if not PREDICT_FLAG:
    #  Read the results from the results json file
    results = load_from_json(results_file)
    model_config = load_from_json(model_config_file)
    # We will save the classifier results and model config for each subtask in this dictionary
    all_subtasks_results_and_model_configs = dict()
    for key in results:
      if key not in ["best_dev_threshold", "best_dev_F1s", "dev_t_F1_P_Rs"]:
        tested_tasks.append(key)
        results[key]["best_dev_threshold"] = results["best_dev_threshold"][key]
        results[key]["best_dev_F1"] = results["best_dev_F1s"][key]
        results[key]["dev_t_F1_P_Rs"] = results["dev_t_F1_P_Rs"][key]
        all_subtasks_results_and_model_configs[key] = results[key], model_config
    all_task_results_and_model_configs[taskname] = all_subtasks_results_and_model_configs
    all_task_question_tags[taskname] = tested_tasks

if not PREDICT_FLAG:
  # Read the results for each task and save them in csv file
  # NOTE: After fixing the USER and URL tags
  results_tsv_save_file = os.path.join("results",
                                       f"all_experiments_multitask_{model_type}_{run_name}_entity_classifier_fixed_results.tsv")
  with open(results_tsv_save_file, "w") as tsv_out:
    writer = csv.writer(tsv_out, delimiter='\t')
    header = ["Event", "Sub-task", "Train Data (size, pos., neg.)", "Dev Data (size, pos., neg.)",
              "Test Data (size, pos., neg.)", "model name", "accuracy", "CM", "pos. F1", "SQuAD_total", "SQuAD_EM",
              "SQuAD_F1", "SQuAD_Pos. EM_F1_total", "SQuAD_Pos. EM", "SQuAD_Pos. F1", "dev_threshold", "dev_N", "dev_F1",
              "dev_P", "dev_R", "dev_TP", "dev_FP", "dev_FN", "N", "F1", "P", "R", "TP", "FP", "FN"]
    writer.writerow(header)
    for taskname, question_tags in all_task_question_tags.items():
      if taskname not in run_tasks:
        continue
      current_task_results_and_model_configs = all_task_results_and_model_configs[taskname]
      for question_tag in question_tags:
        results, model_config = current_task_results_and_model_configs[question_tag]
        # Extract results
        classification_report = results["Classification Report"]
        positive_f1_classification_report = classification_report['1']['f1-score']
        accuracy = classification_report['accuracy']
        CM = results["CM"]
        # SQuAD results
        total_EM = results["SQuAD_EM"]
        total_F1 = results["SQuAD_F1"]
        total_tweets = results["SQuAD_total"]
        pos_EM = results["SQuAD_Pos. EM"]
        pos_F1 = results["SQuAD_Pos. F1"]
        total_pos_tweets = results["SQuAD_Pos. EM_F1_total"]
        # Best threshold and dev F1
        best_dev_threshold = results["best_dev_threshold"]
        best_dev_F1 = results["best_dev_F1"]
        dev_t_F1_P_Rs = results["dev_t_F1_P_Rs"]
        best_dev_threshold_index = int(best_dev_threshold * 10) - 1
        # Each entry in dev_t_F1_P_Rs is of the format t, dev_F1, dev_P, dev_R, dev_TP + dev_FN, dev_TP, dev_FP, dev_FN
        t, dev_F1, dev_P, dev_R, dev_N, dev_TP, dev_FP, dev_FN = dev_t_F1_P_Rs[best_dev_threshold_index]
        # Alan's metrics
        F1 = results["F1"]
        P = results["P"]
        R = results["R"]
        TP = results["TP"]
        FP = results["FP"]
        FN = results["FN"]
        N = results["N"]
        # Extract model config
        model_name = model_config["model"]
        train_data = (
        model_config["train_data"]["size"], model_config["train_data"]["pos"], model_config["train_data"]["neg"])
        dev_data = (model_config["dev_data"]["size"], model_config["dev_data"]["pos"], model_config["dev_data"]["neg"])
        test_data = (
        model_config["test_data"]["size"], model_config["test_data"]["pos"], model_config["test_data"]["neg"])

        row = [taskname, question_tag, train_data, dev_data, test_data, model_name, accuracy, CM,
               positive_f1_classification_report, total_tweets, total_EM, total_F1, total_pos_tweets, pos_EM, pos_F1,
               best_dev_threshold, dev_N, dev_F1, dev_P, dev_R, dev_TP, dev_FP, dev_FN, N, F1, P, R, TP, FP, FN]
        writer.writerow(row)





