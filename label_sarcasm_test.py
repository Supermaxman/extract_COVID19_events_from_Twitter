
import os
import argparse
import json

from tqdm import tqdm


def read_json_lines(path):
	tweets = []
	with open(path, 'r') as f:
		for line in f:
			line = line.strip()
			if line:
				tweet = json.loads(line)
				tweets.append(tweet)
	return tweets


def write_json_lines(data, path):
	with open(path, 'a') as f:
		for i in data:
			line = json.dumps(i)
			f.write(f'{line}\n')


def read_ids(file_path):
	tweet_ids = set()
	num_sarcastic = 0
	with open(file_path, 'r') as f:
		for line in f:
			line = line.strip()
			if line:
				tweet = json.loads(line)
				if tweet_labels['golden_annotation']['part2-sarcasm.Response'][0].lower() == 'yes':
					num_sarcastic += 1
				tweet_ids.add(tweet['id'])
	return tweet_ids, num_sarcastic


parser = argparse.ArgumentParser()

parser.add_argument(
	"-d", "--data_file", help="Path to data file", type=str,
	default='./data/shared_task_test_set_final/shared_task-test-cure.jsonl'
)
parser.add_argument(
	"-l", "--label_file", help="Path to data file", type=str,
	default='./data/shared_task-test_set-eval/cure_sol.jsonl'
)
parser.add_argument(
	"-o", "--output_file", help="Path to output file", type=str,
	default='./data/shared_task-test_set-eval/cure_sol-sarcasm.jsonl'
)

args = parser.parse_args()

data_file = args.data_file
label_file = args.label_file
output_file = args.output_file

seen_ids = set()
if os.path.exists(output_file):
	seen_ids, num_sarcastic = read_ids(output_file)

tweets = read_json_lines(data_file)
labels = read_json_lines(label_file)
labels = {l['id']: l for l in labels}
num_tweets = len(tweets)
print(f'Number of tweets: {num_tweets}')
print(f'Num sarcastic: {num_sarcastic}')

for idx, tweet in enumerate(tweets):
	tweet_id = tweet['id']
	if tweet_id in seen_ids:
		continue
	tweet_labels = labels[tweet_id]
	text = tweet['text']
	print('-----------------------')
	print(f'[{idx+1}/{num_tweets}] Tweet {tweet_id}')
	print('-----------------------')
	print(text)
	print('-----------------------')
	label = 'Not Specified'
	needs_label = True
	while needs_label:
		label = input('Contains sarcasm? (y/n): ')
		label = label.lower().strip()
		if label == 'y':
			label = 'Yes'
			needs_label = False
		elif label == 'n' or not label:
			label = 'Not Specified'
			needs_label = False
		else:
			print(f'Unknown label: {label}')
			needs_label = True
	print('-----------------------')
	tweet_labels['golden_annotation']['part2-sarcasm.Response'] = [label]
	write_json_lines([tweet_labels], output_file)
