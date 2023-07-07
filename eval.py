import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import math
import time
import random
import argparse
from tqdm import tqdm
from functools import partial

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import DebertaTokenizerFast, DebertaForQuestionAnswering
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_

from dataset import TextQADatasetHF, collate_fn_hf
from scheduler import update_lr
from utils import F1EM_score, post_process_prediction, Frame_F1_scores, Frame_F1_score
from utils import align_word_index, approximate_span_time

device = "cuda" if torch.cuda.is_available() else "cpu"
print('device:', device)

# set parameters and arguments
parser = argparse.ArgumentParser()
parser.add_argument("--test-meta-path", 
        type=str, 
        required=True, 
        help="The meta csv file of evaluation data")
parser.add_argument("--LM-path", 
        type=str, 
        required=True, 
        help="The path of pretrained ckpt of language model")
parser.add_argument("--ckpt-path", 
        type=str, 
        required=True, 
        help="The path of QA-trained ckpt of language model")
parser.add_argument("--test-batch-size", 
        type=int, 
        required=True) 
parser.add_argument("--source", 
        type=str, 
        required=True)
args = parser.parse_args()

# fix random seed
from pytorch_lightning import seed_everything
seed_everything(42)

# init tokenizer
tokenizer = DebertaTokenizerFast.from_pretrained(args.LM_path, add_prefix_space=True)

# init dataset and dataloader (train and dev set)
test_dataset = TextQADatasetHF(args.test_meta_path, tokenizer)
test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, collate_fn=partial(collate_fn_hf, tokenizer=tokenizer), shuffle=False, num_workers=4)

# init model and optimizer
model = DebertaForQuestionAnswering.from_pretrained(args.LM_path).to(device)

# load trained weights
print(f'load weights from ckpt : {args.ckpt_path}')
model.load_state_dict(torch.load(args.ckpt_path))



# eval 
print('Testing')
model.eval()
F1s = []
EMs = []
FF1s = []
losses = []
with torch.no_grad():
    with tqdm(test_dataloader, unit='batch') as tbar:
        for batch_idx, batch in enumerate(tbar):
            # preprocess batch
            #inputs = {k:v.to(device) for k, v in batch['inputs'].items()}
            inputs = batch['inputs'].to(device)
            document_start_indices = batch['document start index']
            target_spans = batch['token answer spans']
            word_ids_list = batch['word ids']
            random_target_spans_for_loss = [random.choice(sps) for sps in target_spans] 
            random_target_start_index = torch.tensor([sp[1] for sp, dsi in zip(random_target_spans_for_loss, document_start_indices)]).to(device)
            random_target_end_index = torch.tensor([sp[2] for sp, dsi in zip(random_target_spans_for_loss, document_start_indices)]).to(device)

            # feed into model
            outputs = model(**inputs, start_positions=random_target_start_index, end_positions=random_target_end_index)

            # calculate loss
            loss = outputs.loss

            # process model output
            start_logprob = F.log_softmax(outputs.start_logits, dim=-1)
            end_logprob = F.log_softmax(outputs.end_logits, dim=-1)
            f1s, ems, ff1s = [], [], []
            for i, (start_logp, end_logp, tspans, document_start_index, word2time, word_ids)\
                    in enumerate(zip(start_logprob, end_logprob, target_spans, document_start_indices, batch['word2time'], word_ids_list)):
                # calculate predicted spans
                predicted_start_index, predicted_end_index = post_process_prediction(start_logp[document_start_index+1:], end_logp[document_start_index+1:], 3, 50) 
                
                # calculate start and end time of spans
                # (token -> word -> spans)
                predicted_start_word_index, predicted_end_word_index = align_word_index(word_ids, [predicted_start_index, predicted_end_index])
                #predicted_start_word_index, predicted_end_word_index, tokens = align_word_index(\
                #        input_ids[document_start_index+1:], \
                #        tokenizer, \
                #        pred_span=[predicted_start_index, predicted_end_index],\
                #        document_start_index=document_start_index, \
                #        sep_id=tokenizer.sep_token_id, \
                #        space_token=space_token)
                word2time_entities = word2time[predicted_start_word_index:predicted_end_word_index+1] 
                # get predicted_start_time and predicted_time
                try:
                    predicted_start_time = float(word2time_entities[0][2])
                    predicted_end_time = float(word2time_entities[-1][3])
                except:
                    predicted_start_time, predicted_end_time = approximate_span_time(word2time, [predicted_start_word_index, predicted_end_word_index])
                    #print(word2time_entities)
                    #import pdb;pdb.set_trace()
                    #continue

                # calculate F1 and EM score
                max_f1 = -1
                max_em = -1
                max_ff1 = -1
                for _, target_start_index, target_end_index, target_start_time, target_end_time in tspans:
                    target_start_index -= document_start_index + 1
                    target_end_index -= document_start_index + 1

                    f1, em = F1EM_score(predicted_start_index, predicted_end_index+1, target_start_index, target_end_index+1)
                    ff1 = Frame_F1_score(predicted_start_time, predicted_end_time, target_start_time, target_end_time)
                    max_ff1, max_f1, max_em = max([max_ff1, max_f1, max_em], [ff1, f1, em], key=lambda x:x[0])
                    #import pdb;pdb.set_trace()
                f1s.append(max_f1)
                ems.append(max_em)
                ff1s.append(max_ff1)
            F1 = sum(f1s) / len(f1s)
            EM = sum(ems) / len(ems)
            FF1 = sum(ff1s) / len(ff1s)
            ave_loss = loss.mean().item()
            F1s.append(F1)
            EMs.append(EM)
            FF1s.append(FF1)
            losses.append(ave_loss)

            # record output and scores
            tbar.set_postfix({'loss':round(ave_loss, 2), 'F1':round(F1, 2), 'EM':round(EM, 2), 'FF1':round(FF1,2)})

# calcualte average of scores
ave_F1 = sum(F1s) / len(F1s)
ave_EM = sum(EMs) / len(EMs)
ave_FF1 = sum(FF1s) / len(FF1s)
ave_Loss = sum(losses) / len(losses)
print(f'Average F1: {ave_F1}')
print(f'Average EM: {ave_EM}')
print(f'Average FF1: {ave_FF1}')
print(f'Average loss: {ave_Loss}')

#############################################################
#with open(f'log/verified-test_score/{args.source}.verified.score', 'w+') as f: ####### verified ###
##with open(f'log/test_score/{args.source}.score', 'w+') as f:
#    f.write(f'{args.source}\n')
#    f.write(f'F1:{ave_F1}\n')
#    f.write(f'EM:{ave_EM}\n')
#    f.write(f'FF1:{ave_FF1}\n')
#    f.write(f'loss:{ave_loss}\n')
#############################################################


