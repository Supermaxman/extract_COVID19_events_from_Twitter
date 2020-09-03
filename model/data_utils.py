
from torch.utils.data import Dataset, DataLoader
import torch


import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

Q_TOKEN = "<Q_TARGET>"


class COVID19TaskDataset(Dataset):
	"""COVID19TaskDataset is a generic dataset class which will read data related to different questions"""
	def __init__(self, instances):
		super(COVID19TaskDataset, self).__init__()
		self.instances = instances
		self.nsamples = len(self.instances)

	def __getitem__(self, index):
		return self.instances[index]

	def __len__(self):
		return self.nsamples


class TokenizeCollator():
	def __init__(self, tokenizer, subtasks, entity_start_token_id, entity_end_token_id):
		self.tokenizer = tokenizer
		self.subtasks = subtasks
		self.entity_start_token_id = entity_start_token_id
		self.entity_end_token_id = entity_end_token_id

	def fix_user_mentions_in_tokenized_tweet(self, tokenized_tweet):
		return ' '.join(["@USER" if word.startswith("@") else word for word in tokenized_tweet.split()])

	def __call__(self, batch):
		all_bert_model_input_texts = list()
		gold_labels = {subtask: list() for subtask in self.subtasks}
		cake_ids = list()
		# text :: candidate_chunk :: candidate_chunk_id :: chunk_start_text_id :: chunk_end_text_id :: tokenized_tweet :: tokenized_tweet_with_masked_q_token :: tagged_chunks :: question_label
		for text, chunk, cake_id, chunk_id, chunk_start_text_id, chunk_end_text_id, tokenized_tweet, tokenized_tweet_with_masked_chunk, subtask_labels_dict in batch:
			tokenized_tweet_with_masked_chunk = self.fix_user_mentions_in_tokenized_tweet(tokenized_tweet_with_masked_chunk)
			if chunk in ["AUTHOR OF THE TWEET", "NEAR AUTHOR OF THE TWEET"]:
				# First element of the text will be considered as AUTHOR OF THE TWEET or NEAR AUTHOR OF THE TWEET
				bert_model_input_text = tokenized_tweet_with_masked_chunk.replace(Q_TOKEN, "<E> </E>")
				# print(tokenized_tweet_with_masked_chunk)
				# print(bert_model_input_text)
				# exit()
			else:
				bert_model_input_text = tokenized_tweet_with_masked_chunk.replace(Q_TOKEN, "<E> " + chunk + " </E>")
			all_bert_model_input_texts.append(bert_model_input_text)
			# Add subtask labels in the gold_labels dictionary

			for subtask in self.subtasks:
				gold_labels[subtask].append(subtask_labels_dict[subtask][1])
			cake_ids.append(cake_id)
		# Tokenize
		all_bert_model_inputs_tokenized = self.tokenizer.batch_encode_plus(
			all_bert_model_input_texts,
			# pad_to_max_length=True,
			padding=True,
			return_tensors="pt"
		)
		input_ids, token_type_ids, attention_mask = all_bert_model_inputs_tokenized['input_ids'], all_bert_model_inputs_tokenized['token_type_ids'], all_bert_model_inputs_tokenized['attention_mask']

		# First extract the indices of <E> token in each sentence and save it in the batch
		# TODO try better fusion function like max pooling over span
		# [bsize, 2] where [:, 0] is the batch idx and [:, 1] is the position.
		entity_start_positions = (input_ids == self.entity_start_token_id).nonzero()
		print(entity_start_positions)
		print(entity_start_positions.shape)
		entity_end_positions = (input_ids == self.entity_end_token_id).nonzero()
		print(entity_end_positions)
		print(entity_end_positions.shape)
		entity_span_widths = entity_end_positions[:, 1] - entity_start_positions[:, 1] - 1
		print(entity_span_widths)
		print(entity_span_widths.shape)
		entity_mask = create_mask(entity_start_positions, entity_end_positions, input_ids.shape[1])
		print(entity_mask)
		print(entity_mask.shape)
		exit()
		# Also extract the gold labels
		labels = {subtask: torch.LongTensor(subtask_gold_labels) for subtask, subtask_gold_labels in gold_labels.items()}
		# print(len(batch))
		cake_ids = torch.LongTensor(cake_ids)
		if entity_start_positions.size(0) == 0:
			# Send entity_start_positions to [CLS]'s position i.e. 0
			entity_start_positions = torch.zeros(input_ids.size(0), 2).long()

		if entity_end_positions.size(0) == 0:
			entity_end_positions = torch.zeros(input_ids.size(0), 2).long()

		# print(entity_start_positions)
		# print(input_ids.size(), labels.size())

		# Verify that the number of labels for each subtask is equal to the number of instances
		for subtask in self.subtasks:
			try:
				assert input_ids.size(0) == labels[subtask].size(0)
			except AssertionError:
				logging.error(f"Error Bad batch: Incorrect number of labels given for the batch of size: {len(batch)}")
				logging.error(f"{subtask}, {labels[subtask]}, {labels[subtask].size(0)}")
				exit()
		# assert input_ids.size(0) == labels.size(0)
		return {
			"input_ids": input_ids,
			"entity_start_positions": entity_start_positions,
			"entity_end_positions": entity_end_positions,
			"entity_span_widths": entity_span_widths,
			"gold_labels": labels,
			"batch_data": batch,
			"cake_ids": cake_ids
		}


def create_mask(start_indices, end_indices, seq_len):
	batch_size = start_indices.shape[0]
	cols = torch.LongTensor(range(seq_len)).repeat(batch_size, 1)
	beg = start_indices[:, 1].unsqueeze(1).repeat(1, seq_len)
	end = end_indices[:, 1].unsqueeze(1).repeat(1, seq_len)
	mask = cols.ge(beg) & cols.lt(end)
	# mask.float()
	return mask

