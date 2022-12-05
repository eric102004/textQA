from torch.utils.data.dataset import Dataset
import copy
from utils import align_index

class TextQADataset(Dataset):
    def __init__(self, meta_path, tokenizer):
        self.meta_path = meta_path
        sep_id = tokenizer.sep_token_id
        space_token = tokenizer.convert_ids_to_tokens(tokenizer('hello world')['input_ids'])[2][0]

        # load csv file, convert to dict
        with open(meta_path, 'r+', newline='', encoding='utf-8') as f:
            entities = list(csv.DictReader(f))

        # convert to qd_pairs qid to entity
        unused_keys = ['case id', 'answer', 'start index', 'end index', 
                'Precision of QA model 1', 'Recall of QA model 1', 
                'F1 of QA model 1', 
                'Precision of QA model 2', 'Recall of QA model 2', 
                'F1 of QA model 2', 
                'Precision of QA model 3', 'Recall of QA model 3', 
                'F1 of QA model 3', 
                'rank in semantic search', 'rank in bm25 search'
                ]
        keys = ['source of question', 'question id', 'question', 'document id', 
                'document title', 'document', 'answer spans']
        qd_pairs_dict = {}
        for entity in entities:
            src = entity['source of question']
            qid = entity['question id']
            q = entity['question'].lower()
            did = entity['document id']
            dt = entity['document title'].lower()
            d = entity['document'].lower()
            ans = entity['answer'].lower()
            start_index = int(entity['start index'])
            end_index = int(entity['end index'])
            if qid not in qd_pairs_dict:
                qd_pairs_dict[qid] = copy.deepcopy(entity)
                for k in unused_keys:
                    if k in qd_pairs_dict[qid]:
                        del qd_pairs_dict[qid][k]
                qd_pairs_dict[qid]['answer spans'] = []
            qd_paris_dict[qid]['answer spans'].append([ans, start_index, end_index])
        self.qd_pairs = list(qd_pairs_dict.values())

        span_fail_count, qa_pair_fail_count = 0, 0
        for entities in self.qd_pairs:
            # convert question and document into token ids
            q = entity['question'].lower()
            d = entity['document'].lower()
            inputs = tokenizer(question, document, return_tensors="pt")
            # convert start/end word index to token index
            document_start_index = get_document_start_index(inputs, sep_id)
            word_spans = entity['answer spans']
            raw_target_spans = align_index(inputs['input_ids'][0], tokenizer, word_spans=word_spans, document_start_index=document_start_index, sep_id=sep_id, space_token=space_token)

            # check span fail count
            target_spans = []
            for sp in raw_target_spans:
                if sp[0]==None:
                    span_fail_count += 1
                else:
                    target_spans.append(sp)
            if len(target_spans)==0:
                qa_pair_fail_count += 1

            entities['input ids'] = inputs
            entities['document start index'] = document_start_index
            entities['token answer spans'] = target_spans


    def __len(self):
        return len(self.qd_pairs) 

    def __getitem__(self, index):
        return self.qd_pairs[index]

def collate_fn(entities):
    new_entities = {}
    for k in entities[0].keys():
        new_entities[k] = []
        for entity in entities:
            new_entities[k].append(entity[k])
    return new_entities

def get_document_start_index(input_ids, sep_id):
    document_start_index = 0
    while input_ids['input_ids'][0][document_start_index]!=sep_id:
        document_start_index += 1
    return document_start_index
