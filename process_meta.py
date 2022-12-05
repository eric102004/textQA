import os
import csv
from tqdm import tqdm
import copy

files = ['../QAcheck/filtered_train_top1000_meta.csv', 
        '../QAcheck/filtered_dev_top37374_meta.csv', 
        '../QAcheck/filtered_test_d1_top37374_meta.csv']

def find_span(ans_tokens, doc_tokens):
    spans = []
    start_index = 0
    while start_index+len(ans_tokens)<=len(doc_tokens):
        if ans_tokens==doc_tokens[start_index:start_index+len(ans_tokens)]:
            end_index = start_index + len(ans_tokens) -1
            spans.append((start_index, end_index))
        start_index += 1
    return spans


for file in files:
    filedir, filename = os.path.split(file)
    target_file = os.path.join(filedir, 'processed_'+filename)
    # read file
    print(f'read meta file')
    with open(file, 'r+', newline='', encoding='utf-8') as f:
        entities = list(csv.DictReader(f))
    header = list(entities[0].keys())
        

    # find all possible start/end index in each triplet
    print(f'find all possible start/end index')
    se_index_list = []
    for entity in tqdm(entities):
        ans = entity['answer']
        doc = entity['document']
        # find all start/end index
        ans_tokens = ans.lower().split(' ')
        doc_tokens = doc.lower().split(' ')
        spans = find_span(ans_tokens, doc_tokens)

        if len(spans)==0:
            qid = entity['question id']
            did = entity['document id']
            print(qid, ans, did)
            print(spans)
            print(doc_tokens)
            raise Exception('Cannot find answer span')

        se_index_list.append(spans)

    # get new entities
    print(f'get new entities')
    new_entities = []
    for entity, spans in tqdm(zip(entities, se_index_list)):
        for start_index, end_index in spans:
            new_entity = copy.deepcopy(entity)
            new_entity['start index'] = start_index
            new_entity['end index'] = end_index
            new_entities.append(new_entity)



    # write to new meta files
    print(f'write to new meta file')
    with open(target_file, 'w+', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for entity in tqdm(new_entities):
            writer.writerow(entity)
