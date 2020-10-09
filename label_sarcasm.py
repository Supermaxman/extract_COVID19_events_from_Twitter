
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
	with open(file_path, 'r') as f:
		for line in f:
			line = line.strip()
			if line:
				tweet = json.loads(line)
				tweet_ids.add(tweet['id'])
	return tweet_ids


parser = argparse.ArgumentParser()

parser.add_argument(
	"-d", "--data_file", help="Path to data file", type=str,
	default='./data/cure_and_prevention-add_text.jsonl'
)
parser.add_argument(
	"-o", "--output_file", help="Path to output file", type=str,
	default='./data/cure_and_prevention-add_text-sarcasm.jsonl'
)

args = parser.parse_args()

data_file = args.data_file
output_file = args.output_file

seen_ids = set()
if os.path.exists(output_file):
	seen_ids = read_ids(output_file)

tweets = read_json_lines(data_file)
num_tweets = len(tweets)
print(f'Number of tweets: {num_tweets}')

for idx, tweet in enumerate(tweets):
	tweet_id = tweet['id']
	if tweet_id in seen_ids:
		continue
	text = tweet['text']
	print('-----------------------')
	print(f'[{idx+1}/{num_tweets}] Tweet {tweet_id}')
	print('Text:')
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
	tweet['annotation']['part2-sarcasm.Response'] = [label]
	write_json_lines([tweet], output_file)
