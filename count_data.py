
from model.utils import *
from collections import defaultdict

file_names = ['positive', 'negative', 'can_not_test', 'death', 'cure_and_prevention']

## generate all files
for each_category in file_names:
    input_file = read_json_line('./data/'+each_category+'.jsonl')
    downloaded_files = set([ex['id'] for ex in read_json_line('./data/'+each_category+'-add_text.jsonl')])
    total = 0
    event_count = 0
    slot_stats = defaultdict(int)
    for each_line in input_file:
        if each_line['id'] in downloaded_files:
            chunks = each_line["candidate_chunks_offsets"]
            annotated_chunks = each_line['annotation']
            total += len(chunks)
            for slot_name, slot_chunks in annotated_chunks.items():
                if 'part2' in slot_name:
                    slot_count = len(slot_chunks)
                    if slot_count == 1 and slot_chunks[0] == 'Not Specified' or slot_chunks[0] == 'no_cure' or slot_chunks[0] == 'NO_CONSENSUS':
                        slot_count = 0
                    slot_stats[slot_name.replace('part2-', '').replace('.Response', '')] += slot_count
                if 'part1' in slot_name:
                    if slot_chunks[0].lower() == 'yes':
                        event_count += 1
    print('--------------')
    print(f'{each_category:<20} candidate slots {total:<8}')
    print(f'{each_category:<20} event count {event_count:<8}')
    print('--------------')
    for slot_name, slot_stat in slot_stats.items():
        print(f'{slot_name:<20} {slot_stat:<8}')
    print('--------------')
