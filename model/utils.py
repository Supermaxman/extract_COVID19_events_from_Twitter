# We will add all the common utility functions over here

import os
import re
import string
import collections
import json
import pickle
import torch

import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

MIN_POS_SAMPLES_THRESHOLD = 10
Q_TOKEN = "<Q_TARGET>"
URL_TOKEN = "<URL>"


def print_list(l):
	for e in l:
		print(e)
	print()


def log_list(l):
	for e in l:
		logging.info(e)
	logging.info("")


def save_in_pickle(save_object, save_file):
	with open(save_file, "wb") as pickle_out:
		pickle.dump(save_object, pickle_out)


def load_from_pickle(pickle_file):
	with open(pickle_file, "rb") as pickle_in:
		return pickle.load(pickle_in)


def save_in_json(save_dict, save_file):
	with open(save_file, 'w') as fp:
		json.dump(save_dict, fp)


def load_from_json(json_file):
	with open(json_file, 'r') as fp:
		return json.load(fp)


def read_json_line(path):

	output = []
	with open(path, 'r') as f:
		for line in f:
			output.append(json.loads(line))

	return output


def write_json_line(data, path):

	with open(path, 'w') as f:
		for i in data:
			f.write("%s\n" % json.dumps(i))

	return None


def make_dir_if_not_exists(directory):
	if not os.path.exists(directory):
		logging.info("Creating new directory: {}".format(directory))
		os.makedirs(directory)


def extract_instances_for_current_subtask(task_instances, sub_task):
	return task_instances[sub_task]


def get_multitask_instances_for_valid_tasks(task_instances, tag_statistics, has_labels=True):
	# Extract instances and labels from all the sub-task
	# Align w.r.t. instances and merge all the sub-task labels
	subtasks = list()
	for subtask in task_instances.keys():
		current_question_tag_statistics = tag_statistics[0][subtask]
		if len(current_question_tag_statistics) > 1 and current_question_tag_statistics[1] >= MIN_POS_SAMPLES_THRESHOLD:
			subtasks.append(subtask)
	
	# For each tweet we will first extract all its instances from each task and their corresponding labels
	text_to_subtask_instances = dict()
	original_text_list = list()
	instance_id = 0
	instances = {}
	for subtask in subtasks:
		# get the instances for current subtask and add it to a set
		for instance in task_instances[subtask]:
			# text, chunk, chunk_id, chunk_start_text_id, chunk_end_text_id, tokenized_tweet, tokenized_tweet_with_masked_chunk, gold_chunk, label
			# instance = (text, chunk, chunk_id, chunk_start_text_id, chunk_end_text_id, tokenized_tweet, tokenized_tweet_with_masked_chunk)
			instance['instance_id'] = instance_id
			instances[instance_id] = instance
			text = instance['text']
			if text not in text_to_subtask_instances:
				original_text_list.append(text)
				text_to_subtask_instances[text] = dict()
			text_to_subtask_instances[text].setdefault(instance_id, dict())
			if has_labels:
				gold_chunk = instance['gold_chunk']
				label = instance['label']
			else:
				gold_chunk = None
				label = None
			text_to_subtask_instances[text][instance_id][subtask] = (gold_chunk, label)
			instance_id += 1
	# print(len(text_to_subtask_instances))
	# print(len(original_text_list))
	# print(sum(len(instance_dict) for text, instance_dict in text_to_subtask_instances.items()))
	# For each instance we need to make sure that it has all the subtask labels
	# For missing subtask labels we will give a default label of 0
	for text in original_text_list:
		for instance_id, subtasks_labels_dict in text_to_subtask_instances[text].items():
			for subtask in subtasks:
				if subtask not in subtasks_labels_dict:
					# Adding empty label for this instance
					subtasks_labels_dict[subtask] = ([], 0)
			# update the subtask labels_dict in the text_to_subtask_instances data structure
			assert len(subtasks_labels_dict) == len(subtasks)
			text_to_subtask_instances[text][instance_id] = subtasks_labels_dict

	# Merge all the instances into one list
	all_multitask_instances = list()
	for text in original_text_list:
		for instance_id, subtasks_labels_dict in text_to_subtask_instances[text].items():
			instance = instances[instance_id]
			instance['subtasks_labels_dict'] = subtasks_labels_dict
			all_multitask_instances.append(instance)
	return all_multitask_instances, subtasks


def split_multitask_instances_in_train_dev_test(multitask_instances, TRAIN_RATIO = 0.6, DEV_RATIO = 0.15):
	# Group the multitask_instances by original tweet
	original_tweets = dict()
	original_tweets_list = list()
	# text :: candidate_chunk :: candidate_chunk_id :: chunk_start_text_id :: chunk_end_text_id :: tokenized_tweet :: tokenized_tweet_with_masked_q_token :: tagged_chunks :: question_label
	for example in multitask_instances:
		tweet = example['text']
		if tweet not in original_tweets:
			original_tweets[tweet] = 1
			original_tweets_list.append(tweet)
		else:
			original_tweets[tweet] += 1
	# print(f"Number of multitask_instances: {len(multitask_instances)} \tNumber of tweets: {len(original_tweets_list)}\t Avg chunks per tweet:{float(len(multitask_instances))/float(len(original_tweets_list))}")
	train_size = int(len(original_tweets_list) * TRAIN_RATIO)
	dev_size = int(len(original_tweets_list) * DEV_RATIO)
	train_tweets = original_tweets_list[:train_size]
	dev_tweets = original_tweets_list[train_size:train_size + dev_size]
	test_tweets = original_tweets_list[train_size + dev_size:]
	segment_multitask_instances = {"train": list(), "dev": list(), "test": list()}
	# A dictionary that stores the segment each tweet belongs to
	tweets_to_segment = dict()
	for tweet in train_tweets:
		tweets_to_segment[tweet] = "train"
	for tweet in dev_tweets:
		tweets_to_segment[tweet] = "dev"
	for tweet in test_tweets:
		tweets_to_segment[tweet] = "test"
	# Get multitask_instances
	for instance in multitask_instances:
		tweet = instance['text']
		segment_multitask_instances[tweets_to_segment[tweet]].append(instance)

	# print(f"Train:{len(train_tweets)}\t Dev:{len(dev_tweets)}\t Test:{len(test_tweets)}")
	# print(f"Train:{len(segment_multitask_instances['train'])}\t Dev:{len(segment_multitask_instances['dev'])}\t Test:{len(segment_multitask_instances['test'])}")
	return segment_multitask_instances['train'], segment_multitask_instances['dev'], segment_multitask_instances['test']

def split_instances_in_train_dev_test(instances, TRAIN_RATIO = 0.6, DEV_RATIO = 0.15):
	# Group the instances by original tweet
	original_tweets = dict()
	original_tweets_list = list()
	# text :: candidate_chunk :: candidate_chunk_id :: chunk_start_text_id :: chunk_end_text_id :: tokenized_tweet :: tokenized_tweet_with_masked_q_token :: tagged_chunks :: question_label
	for example in instances:
		tweet = example['text']
		if tweet not in original_tweets:
			original_tweets[tweet] = 1
			original_tweets_list.append(tweet)
		else:
			original_tweets[tweet] += 1
	# print(f"Number of instances: {len(instances)} \tNumber of tweets: {len(original_tweets_list)}\t Avg chunks per tweet:{float(len(instances))/float(len(original_tweets_list))}")
	train_size = int(len(original_tweets_list) * TRAIN_RATIO)
	dev_size = int(len(original_tweets_list) * DEV_RATIO)
	train_tweets = original_tweets_list[:train_size]
	dev_tweets = original_tweets_list[train_size:train_size + dev_size]
	test_tweets = original_tweets_list[train_size + dev_size:]
	segment_instances = {"train": list(), "dev": list(), "test": list()}
	# A dictionary that stores the segment each tweet belongs to
	tweets_to_segment = dict()
	for tweet in train_tweets:
		tweets_to_segment[tweet] = "train"
	for tweet in dev_tweets:
		tweets_to_segment[tweet] = "dev"
	for tweet in test_tweets:
		tweets_to_segment[tweet] = "test"
	# Get instances
	for instance in instances:
		tweet = instance['text']
		segment_instances[tweets_to_segment[tweet]].append(instance)

	# print(f"Train:{len(train_tweets)}\t Dev:{len(dev_tweets)}\t Test:{len(test_tweets)}")
	# print(f"Train:{len(segment_instances['train'])}\t Dev:{len(segment_instances['dev'])}\t Test:{len(segment_instances['test'])}")
	return segment_instances['train'], segment_instances['dev'], segment_instances['test']


def log_data_statistics(data):
	logging.info(f"Total instances in the data = {len(data)}")
	pos_count = sum(example['label'] for example in data)
	logging.info(f"Positive labels = {pos_count} Negative labels = {len(data) - pos_count}")
	return len(data), pos_count, (len(data) - pos_count)


# SQuAD F-1 evaluation
def normalize_answer(s):
	"""Lower text and remove punctuation, articles and extra whitespace."""
	def remove_articles(text):
		regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
		return re.sub(regex, ' ', text)

	def white_space_fix(text):
		return ' '.join(text.split())

	def remove_punc(text):
		exclude = set(string.punctuation)
		return ''.join(ch for ch in text if ch not in exclude)

	def lower(text):
		return text.lower()
	return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
	if not s: return []
	return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
	return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
	gold_toks = get_tokens(a_gold)
	pred_toks = get_tokens(a_pred)
	common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
	num_same = sum(common.values())
	if len(gold_toks) == 0 or len(pred_toks) == 0:
		# If either is no-answer, then F1 is 1 if they agree, 0 otherwise
		return int(gold_toks == pred_toks)
	if num_same == 0:
		return 0
	precision = 1.0 * num_same / len(pred_toks)
	recall = 1.0 * num_same / len(gold_toks)
	f1 = (2 * precision * recall) / (precision + recall)
	return f1


def get_raw_scores(data, prediction_scores, positive_only=False):
	predicted_chunks_for_each_instance = dict()
	# text :: candidate_chunk :: candidate_chunk_id :: chunk_start_text_id :: chunk_end_text_id :: tokenized_tweet :: tokenized_tweet_with_masked_q_token :: tagged_chunks :: question_label
	#(text, chunk, chunk_id, chunk_start_text_id, chunk_end_text_id, tokenized_tweet, tokenized_tweet_with_masked_chunk, gold_chunk, label)
	for example, prediction_score in zip(data, prediction_scores):
		original_text = example['text']
		gold_chunk = example['gold_chunk']
		chunk = example['chunk']
		# print(text)
		# print(chunk)
		# print(original_text)
		# exit()
		predicted_chunks_for_each_instance.setdefault(original_text, ('', 0.0, set(), set()))
		current_predicted_chunk, current_predicted_chunk_score, predicted_chunks, gold_chunks = predicted_chunks_for_each_instance[original_text]
		if gold_chunk != ['Not Specified']:
			gold_chunks = gold_chunks.union(set(gold_chunk))
		if prediction_score > 0.5:
			# Save this prediction in the predicted chunks
			predicted_chunks.add(chunk)
			current_predicted_chunk_score = prediction_score
			current_predicted_chunk = chunk
			predicted_chunks_for_each_instance[original_text] = current_predicted_chunk, current_predicted_chunk_score, predicted_chunks, gold_chunks
		elif prediction_score > current_predicted_chunk_score:
			# only update the current_predicted_chunk and its score
			current_predicted_chunk_score = prediction_score
			current_predicted_chunk = chunk
			predicted_chunks_for_each_instance[original_text] = current_predicted_chunk, current_predicted_chunk_score, predicted_chunks, gold_chunks
	total = 0.0
	exact_scores, f1_scores = 0.0, 0.0
	for original_text, (current_predicted_chunk, current_predicted_chunk_score, predicted_chunks, gold_chunks) in predicted_chunks_for_each_instance.items():
		if len(gold_chunks) > 0:
			if len(predicted_chunks) > 0:
				for predicted_chunk in predicted_chunks:
					# Get best exact_score and f1_score compared to all the gold_chunks
					best_exact_score, best_f1_score = 0.0, 0.0
					for gold_chunk in gold_chunks:
						best_exact_score = max(best_exact_score, compute_exact(gold_chunk, predicted_chunk))
						best_f1_score = max(best_f1_score, compute_f1(gold_chunk, predicted_chunk))
					exact_scores += best_exact_score
					f1_scores += best_f1_score
					total += 1.0
			else:
				# Assume the top prediction for this (text, gold_chunk) pair as final prediction
				# Get best exact_score and f1_score compared to all the gold_chunks
				best_exact_score, best_f1_score = 0.0, 0.0
				# for gold_chunk in gold_chunks:
				# 	best_exact_score = max(best_exact_score, compute_exact(gold_chunk, current_predicted_chunk))
				# 	best_f1_score = max(best_f1_score, compute_f1(gold_chunk, current_predicted_chunk))
				exact_scores += best_exact_score
				f1_scores += best_f1_score
				total += 1.0
			
			# exact_scores += compute_exact(gold_chunk, current_predicted_chunk)
			# f1_scores += compute_f1(gold_chunk, current_predicted_chunk)
		elif len(gold_chunks) == 0 and not positive_only:
			if len(predicted_chunks) > 0:
				# Model predicted something and the gold is also nothing
				for i in range(len(predicted_chunks)):
					# Penalize for every incorrect predicted chunk
					best_exact_score, best_f1_score = 0.0, 0.0
					exact_scores += best_exact_score
					f1_scores += best_f1_score
					total += 1.0
			else:
				# Model predicted nothing and the gold is also nothing
				best_exact_score, best_f1_score = 1.0, 1.0
				exact_scores += best_exact_score
				f1_scores += best_f1_score
				total += 1.0
			# exact_scores += compute_exact(gold_chunk, current_predicted_chunk)
			# f1_scores += compute_f1(gold_chunk, current_predicted_chunk)
		
	if total == 0:
		predictions_exact_score = total
		predictions_f1_score = total
	else:
		predictions_exact_score = exact_scores * 100.0 / total
		predictions_f1_score = f1_scores * 100.0 / total
	return predictions_exact_score, predictions_f1_score, total


def get_threshold_predictions(data, prediction_scores, THRESHOLD=0.5):
	predicted_chunks_for_each_instance = dict()
	assert len(data) == len(prediction_scores)
	for example, prediction_score in zip(data, prediction_scores):
		doc_id = example['doc_id']
		original_chunk = example['text'][example['chunk_start_text_id']:example['chunk_end_text_id']]
		candidate_chunk = example['chunk']
		chunk = original_chunk if candidate_chunk != 'AUTHOR OF THE TWEET' else candidate_chunk
		# print(text)
		# print(chunk)
		# print(original_text)
		# exit()
		predicted_chunks_for_each_instance.setdefault(doc_id, ('', 0.0, set()))
		current_predicted_chunk, current_predicted_chunk_score, predicted_chunks = \
			predicted_chunks_for_each_instance[doc_id]

		if prediction_score > THRESHOLD:
			# Save this prediction in the predicted chunks
			predicted_chunks.add(chunk)
			current_predicted_chunk_score = prediction_score
			current_predicted_chunk = chunk
			predicted_chunks_for_each_instance[doc_id] = current_predicted_chunk, current_predicted_chunk_score, predicted_chunks
		elif prediction_score > current_predicted_chunk_score:
			# only update the current_predicted_chunk and its score
			current_predicted_chunk_score = prediction_score
			current_predicted_chunk = chunk
			predicted_chunks_for_each_instance[doc_id] = current_predicted_chunk, current_predicted_chunk_score, predicted_chunks

	doc_predicted_chunks = {}
	for doc_id, (_, _, doc_pred_chunks) in predicted_chunks_for_each_instance.items():
		doc_predicted_chunks[doc_id] = doc_pred_chunks
	return doc_predicted_chunks


def get_TP_FP_FN(data, prediction_scores, THRESHOLD=0.5):
	predicted_chunks_for_each_instance = dict()
	# (text, chunk, chunk_id, chunk_start_text_id, chunk_end_text_id, tokenized_tweet, tokenized_tweet_with_masked_chunk, gold_chunk, label)
	for example, prediction_score in zip(data, prediction_scores):
		original_text = example['text']
		gold_chunk = example['gold_chunk']
		label = example['label']
		chunk = example['chunk']
		# print(text)
		# print(chunk)
		# print(original_text)
		# exit()
		predicted_chunks_for_each_instance.setdefault(original_text, ('', 0.0, set(), set()))
		current_predicted_chunk, current_predicted_chunk_score, predicted_chunks, gold_chunks = predicted_chunks_for_each_instance[original_text]
		# if label == 1:
		# 	print(gold_chunk)
		# 	print(gold_chunks)
		if gold_chunk != ['Not Specified'] and label == 1:
			gold_chunks = gold_chunks.union(set(gold_chunk))
			predicted_chunks_for_each_instance[original_text] = current_predicted_chunk, current_predicted_chunk_score, predicted_chunks, gold_chunks
		if prediction_score > THRESHOLD:
			# Save this prediction in the predicted chunks
			predicted_chunks.add(chunk)
			current_predicted_chunk_score = prediction_score
			current_predicted_chunk = chunk
			predicted_chunks_for_each_instance[original_text] = current_predicted_chunk, current_predicted_chunk_score, predicted_chunks, gold_chunks
		elif prediction_score > current_predicted_chunk_score:
			# only update the current_predicted_chunk and its score
			current_predicted_chunk_score = prediction_score
			current_predicted_chunk = chunk
			predicted_chunks_for_each_instance[original_text] = current_predicted_chunk, current_predicted_chunk_score, predicted_chunks, gold_chunks
	
	# Take every span that is predicted by the model, and every gold span in the data.
	# Then, you can easily calculate the number of True Positives, False Positives and False negatives.
	# True positives are predicted spans that appear in the gold labels.
	# False positives are predicted spans that don't appear in the gold labels.
	# False negatives are gold spans that weren't in the set of spans predicted by the model.
	# then you can compute P/R using the standard formulas: P= TP/(TP + FP). R = TP/(TP+FN)

	TP, FP, FN = 0.0, 0.0, 0.0
	total_gold_chunks = 0
	for original_text, (current_predicted_chunk, current_predicted_chunk_score, predicted_chunks, gold_chunks) in predicted_chunks_for_each_instance.items():
		total_gold_chunks += len(gold_chunks)
		if len(gold_chunks) > 0:
			# print(f"{len(gold_chunks)} Gold chunks: {gold_chunks} for tweet {original_text}")
			if len(predicted_chunks) > 0:
				for predicted_chunk in predicted_chunks:
					if predicted_chunk in gold_chunks:
						TP += 1		# True positives are predicted spans that appear in the gold labels.
					else:
						FP += 1		# False positives are predicted spans that don't appear in the gold labels.
			for gold_chunk in gold_chunks:
				if gold_chunk not in predicted_chunks:
					FN += 1			# False negatives are gold spans that weren't in the set of spans predicted by the model.
		else:
			if len(predicted_chunks) > 0:
				for predicted_chunk in predicted_chunks:
					FP += 1			# False positives are predicted spans that don't appear in the gold labels.


	if TP + FP == 0:
		P = 0.0
	else:
		P = TP / (TP + FP)
	
	if TP + FN == 0:
		R = 0.0
	else:
		R = TP / (TP + FN)

	if P + R == 0:
		F1 = 0.0
	else:
		F1 = 2.0 * P * R / (P + R)
	return F1, P, R, TP, FP, FN


def make_dir_if_not_exists(directory):
	if not os.path.exists(directory):
		logging.info("Creating new directory: {}".format(directory))
		os.makedirs(directory)


def format_time(elapsed):
	'''
	Takes a time in seconds and returns a string hh:mm:ss
	'''
	# Round to the nearest second.
	elapsed_rounded = int(round((elapsed)))

	# Format as hh:mm:ss
	return str(datetime.timedelta(seconds=elapsed_rounded))


def plot_train_loss(loss_trajectory_per_epoch, trajectory_file):
	plt.cla()
	plt.clf()

	fig, ax = plt.subplots()
	x = [epoch * len(loss_trajectory) + j + 1 for epoch, loss_trajectory in enumerate(loss_trajectory_per_epoch) for j, avg_loss in enumerate(loss_trajectory) ]
	x_ticks = [ "(" + str(epoch + 1) + "," + str(j + 1) + ")" for epoch, loss_trajectory in enumerate(loss_trajectory_per_epoch) for j, avg_loss in enumerate(loss_trajectory) ]
	full_train_trajectory = [avg_loss for epoch, loss_trajectory in enumerate(loss_trajectory_per_epoch) for j, avg_loss in enumerate(loss_trajectory)]
	ax.plot(x, full_train_trajectory)

	ax.set(xlabel='Epoch, Step', ylabel='Loss',
			title='Train loss trajectory')
	step_size = 100
	ax.xaxis.set_ticks(range(0, len(x_ticks), step_size), x_ticks[::step_size])
	ax.grid()

	fig.savefig(trajectory_file)


def split_data_based_on_subtasks(data, subtasks, has_labels=True):
	# We will split the data into data_instances based on subtask_labels
	subtasks_data = {subtask: list() for subtask in subtasks}
	# text, chunk, chunk_id, chunk_start_text_id, chunk_end_text_id, tokenized_tweet, tokenized_tweet_with_masked_chunk, subtasks_labels_dict
	for example in data:
		for subtask in subtasks:
			# text, chunk, chunk_id, chunk_start_text_id, chunk_end_text_id, tokenized_tweet, tokenized_tweet_with_masked_chunk,
			subtask_example = example.copy()
			if has_labels:
				subtask_label = subtask_example['subtasks_labels_dict'][subtask]
				subtask_example['gold_chunk'] = subtask_label[0]
				subtask_example['label'] = subtask_label[1]
			subtasks_data[subtask].append(subtask_example)
	return subtasks_data


def log_multitask_data_statistics(data, subtasks):
	logging.info(f"Total instances in the data = {len(data)}")
	# print positive and negative counts for each subtask
	# print(len(data[0]))
	pos_counts = {subtask: sum(example['subtasks_labels_dict'][subtask][1] for example in data) for subtask in subtasks}
	# Log for each subtask
	neg_counts = dict()
	for subtask in subtasks:
		neg_counts[subtask] = len(data) - pos_counts[subtask]
		logging.info(f"Subtask:{subtask:>15}\tPositive labels = {pos_counts[subtask]}\tNegative labels = {neg_counts[subtask]}")
	return len(data), pos_counts, neg_counts


def get_optimizer_params(model, weight_decay):
	param_optimizer = list(model.named_parameters())
	no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
	optimizer_params = [
		{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
		 'weight_decay': weight_decay},
		{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

	return optimizer_params


def create_mask(start_indices, end_indices, seq_len):
	batch_size = start_indices.shape[0]
	cols = torch.LongTensor(range(seq_len)).repeat(batch_size, 1)
	beg = start_indices[:, 1].unsqueeze(1).repeat(1, seq_len)
	end = end_indices[:, 1].unsqueeze(1).repeat(1, seq_len)
	mask = cols.ge(beg) & cols.lt(end)
	mask = mask.float()
	return mask
