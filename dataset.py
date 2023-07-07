from torch.utils.data.dataset import Dataset
import csv
import copy
from utils import align_index, get_document_start_index
import torch
from tqdm import tqdm
import os
import json
import random
from datasets import load_dataset



class TextQADatasetHF(Dataset):
    def __init__(self, meta_path, tokenizer, sqa5_dir='../slue-sqa5'):
        with open(meta_path, 'r+', encoding='utf-8') as f:
            self.reader = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
        
        self.document_start_index_list = []
        self.token_answer_spans_list = []
        for idx, row in tqdm(enumerate(self.reader), total=len(self.reader)):
            self.reader[idx]['answer_spans'] = json.loads(row['answer_spans'])
            q_words = row['normalized_question_text'].lower().split()
            d_words = row['normalized_document_text'].lower().split()
            inputs = tokenizer(q_words, d_words, return_tensors="pt", is_split_into_words=True)
            # getting document start index
            document_start_index = get_document_start_index(inputs, tokenizer.sep_token_id)
            self.document_start_index_list.append(document_start_index)
            # getting word2time
            split = os.path.basename(meta_path)[:-4].split('_')[1]
            word2time = load_word2time(os.path.join(sqa5_dir, split, 'word2time', row['document_id']+'.txt'))
            self.reader[idx]['word2time'] = word2time
            # getting word answer spans
            word_spans =  []
            for answer_span in self.reader[idx]['answer_spans']:
                start_idx, end_idx = None, None
                word_idx = -1
                cur_word = None
                for idx, w2t in enumerate(word2time):
                    if len(w2t)==1:
                        continue
                    if cur_word!= w2t[0]:
                        word_idx += 1
                        cur_word = w2t[0]
                    if answer_span[1] == float(w2t[2]) and start_idx==None:
                        start_idx = idx
                        start_word_idx = word_idx
                    if answer_span[2] == float(w2t[3]) and end_idx==None:
                        end_idx = idx
                        end_word_idx = word_idx
                    if start_idx!=None and end_idx!=None:
                        break
                #if not(start_idx!=None and end_idx!=None):
                #    import pdb;pdb.set_trace()
                assert start_idx!=None and end_idx!=None
                word_spans.append([answer_span[0], start_word_idx, end_word_idx, answer_span[1], answer_span[2]])
            # getting token answer spans
            token_answer_spans = align_index(inputs, word_spans, tokenizer)
            self.token_answer_spans_list.append(token_answer_spans)


    def __len__(self):
        return len(self.reader)

    def __getitem__(self, index):
        entity = self.reader[index]
        entity['document start index'] = self.document_start_index_list[index]
        entity['token answer spans'] = self.token_answer_spans_list[index]
        return entity

def collate_fn_hf(entities, tokenizer):
    new_entities = {}
    for k in entities[0].keys():
        new_entities[k] = []
        for entity in entities:
            new_entities[k].append(entity[k])
    # getting tokenized inputs
    inputs = tokenizer(new_entities['normalized_question_text'], new_entities['normalized_document_text'], padding=True, truncation=True, return_tensors='pt')
    new_entities['inputs'] = inputs
    new_entities['word ids'] = [inputs.word_ids(i) for i in range(len(entities))]
    return new_entities

def load_word2time(word2time_file):
    word2time = []
    with open(word2time_file, 'r+') as f:
        for line in f.readlines():
            word2time.append(line.strip().split('\t'))
    return word2time

