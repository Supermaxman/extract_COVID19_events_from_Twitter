
from model.utils import read_json_line
import json

task_type_to_datapath_dict = {
  "tested_positive": ("./data/positive-add_text.jsonl", "./data/positive_ctxt2id.json"),
  "tested_negative": ("./data/negative-add_text.jsonl", "./data/test_negative_ctxt2id.json"),
  "can_not_test": ("./data/can_not_test-add_text.jsonl", "./data/can_not_test_ctxt2id.json"),
  "death": ("./data/death-add_text.jsonl", "./data/death_ctxt2id.json"),
  "cure": ("./data/cure_and_prevention-add_text.jsonl", "./data/cure_and_prevention_ctxt2id.json"),
  }

for task, (input_path, output_path) in task_type_to_datapath_dict.items():
  ctxt2id = {}
  cid2id = {}
  examples = read_json_line(input_path)
  for example in examples:
    ex_id = example['id']
    ex_txt = example['text']
    for (start, end) in example['candidate_chunks_offsets']:
      cid = (ex_id, start, end)
      c_txt = ex_txt[start:end]
      ctxt2id[c_txt] = len(ctxt2id)
      cid2id[cid] = ctxt2id[c_txt]
  with open(output_path, 'w') as f:
    json.dump(ctxt2id, f)
