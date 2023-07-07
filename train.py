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
from torch.utils.tensorboard.writer import SummaryWriter

from dataset import TextQADatasetHF, collate_fn_hf
from scheduler import update_lr
from utils import F1EM_score, post_process_prediction, Frame_F1_scores, Frame_F1_score

device = "cuda" if torch.cuda.is_available() else "cpu"
print('device:', device)

# set parameters and arguments
parser = argparse.ArgumentParser()
parser.add_argument("--train-meta-path", 
        type=str, 
        required=True, 
        help="The meta csv file of train set")
parser.add_argument("--test-meta-path", 
        type=str, 
        required=True, 
        help="The meta csv file of test set")
parser.add_argument("--LM-path", 
        type=str, 
        required=True, 
        help="The path of pretrained ckpt of language model")
parser.add_argument("--lr", 
        default=3e-5,
        type=float, 
        help="The max learning rate for Adam optimizer."
        )
parser.add_argument("--max-grad-norm", 
        default=1.0, 
        type=float, 
        help="Max gradient norm for gradient clipping")
parser.add_argument("--warmup-steps", 
        default=100, 
        type=int, 
        help="Linear warmup over warmup_steps")
parser.add_argument("--n-epoch", 
        default=10, 
        type=int, 
        help="Total number of training epochs")
parser.add_argument("--ckpt-dir", 
        type=str, 
        required=True, 
        help="The output directory where the model checkpoints will be saved")
parser.add_argument("--log-dir", 
        type=str, 
        required=True, 
        help="The output directory where the tensorboard event file will be saved")
parser.add_argument("--train-batch-size", 
        type=int, 
        required=True) 
parser.add_argument("--test-batch-size", 
        type=int, 
        required=True) 
parser.add_argument("--gradient-accumulation-steps", 
        default=1,
        type=int)
parser.add_argument("--source", 
        type=str, 
        required=True)
parser.add_argument("--mode",
        type=str, 
        required=True
        )
args = parser.parse_args()
assert args.mode in ['train', 'evaluate']

# init log_dir and ckpt_dir
subdir = time.strftime("%Y%m%d_%H%M%S", time.localtime())
print(f'save checkpoint and log at subdir: {subdir}')
ckpt_dir = os.path.join(args.ckpt_dir, subdir)
log_dir = os.path.join(args.log_dir, subdir)

# fix random seed
from pytorch_lightning import seed_everything
seed_everything(42)

# init tokenizer
tokenizer = DebertaTokenizerFast.from_pretrained(args.LM_path, add_prefix_space=True)

# init dataset and dataloader (train and dev set)
train_dataset = TextQADatasetHF(args.train_meta_path, tokenizer)
test_dataset = TextQADatasetHF(args.test_meta_path, tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, collate_fn=partial(collate_fn_hf, tokenizer=tokenizer), shuffle=True, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, collate_fn=partial(collate_fn_hf, tokenizer=tokenizer), shuffle=False, num_workers=4)

# calculate total steps
total_steps = math.ceil(len(train_dataloader)/args.gradient_accumulation_steps)*args.n_epoch

# init model and optimizer
model = DebertaForQuestionAnswering.from_pretrained(args.LM_path).to(device)
optimizer = Adam(model.parameters(), lr=args.lr, eps=1e-6, weight_decay=0)


# load pretrined weights (for training)
#model.deberta.load_state_dict(torch.load(args.LM_path))

# init tensorboard
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)


space_token = tokenizer.convert_ids_to_tokens(tokenizer('hello world')['input_ids'])[2][0]



# start training
best_FF1 = -1
cur_step = 1
for n in range(args.n_epoch):
    print(f'epoch:{n}')
    print('Training')
    model.train()
    F1s = []
    EMs = []
    losses = []
    with tqdm(train_dataloader, unit='batch') as tbar:
        for batch_idx, batch in enumerate(tbar):

            # preprocess batch
            #inputs = {k:v.to(device) for k, v in batch['inputs'].items()}
            inputs = batch['inputs'].to(device)
            document_start_indices = batch['document start index']
            target_spans = batch['token answer spans']
            random_target_spans_for_loss = [random.choice(sps) for sps in target_spans] 
            random_target_start_index = torch.tensor([sp[1] for sp, dsi in zip(random_target_spans_for_loss, document_start_indices)]).to(device)
            random_target_end_index = torch.tensor([sp[2] for sp, dsi in zip(random_target_spans_for_loss, document_start_indices)]).to(device)
            # feed into model
            outputs = model(**inputs, start_positions=random_target_start_index, end_positions=random_target_end_index)

            # calculate loss
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps 

            # backward
            loss.backward()

            # process model output
            start_logprob = F.log_softmax(outputs.start_logits, dim=-1)
            end_logprob = F.log_softmax(outputs.end_logits, dim=-1)
            f1s, ems = [], []
            for start_logp, end_logp, tspans, document_start_index in \
                    zip(start_logprob, end_logprob, target_spans, document_start_indices):
                # calculate predicted spans
                predicted_start_index, predicted_end_index = post_process_prediction(start_logp[document_start_index+1:], end_logp[document_start_index+1:], 3, 50) 
                # calculate F1 and EM score
                max_f1 = -1
                max_em = -1
                for _, target_start_index, target_end_index, _, _ in tspans:
                    target_start_index -= document_start_index + 1
                    target_end_index -= document_start_index + 1

                    f1, em = F1EM_score(predicted_start_index, predicted_end_index+1, target_start_index, target_end_index+1)
                    max_f1, max_em = max([max_f1, max_em], [f1, em], key=lambda x:x[0])
                f1s.append(max_f1)
                ems.append(max_em)
            F1 = sum(f1s) / len(f1s)
            EM = sum(ems) / len(ems)
            ave_loss = loss.mean().item()
            F1s.append(F1)
            EMs.append(EM)
            losses.append(ave_loss)

            if ((batch_idx+1) % args.gradient_accumulation_steps==0) or (batch_idx+1==len(train_dataloader)):
                # clip gradient
                clip_grad_norm_(model.parameters(), args.max_grad_norm) 
                # update model params and then learning rate
                update_lr(optimizer, args.lr, cur_step, total_steps, warmup_step=args.warmup_steps)
                optimizer.step()
                optimizer.zero_grad()
                cur_step += 1
                
            # record output and scores
            #tbar.set_postfix({'loss':round(ave_loss, 2), 'F1':round(F1, 2)})
            tbar.set_postfix({'loss':round(ave_loss, 2), 'F1':round(F1, 2), 'EM':round(EM, 2)})

    # calcualte average of scores
    ave_F1 = sum(F1s) / len(F1s)
    ave_EM = sum(EMs) / len(EMs)
    ave_Loss = sum(losses) / len(losses)
    print(f'Average F1: {ave_F1}')
    print(f'Average EM: {ave_EM}')
    print(f'Average loss: {ave_Loss}')
    writer.add_scalar('F1/train', ave_F1, n)
    writer.add_scalar('EM/train', ave_EM, n)
    writer.add_scalar('Loss/train', ave_Loss, n)

    # eval on dev set
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
                inputs = {k:v.to(device) for k, v in batch['inputs'].items()}
                document_start_indices = batch['document start index']
                target_spans = batch['token answer spans']
                random_target_spans_for_loss = [random.choice(sps) for sps in target_spans] 
                random_target_start_index = torch.tensor([sp[1] for sp, dsi in zip(random_target_spans_for_loss, document_start_indices)]).to(device)
                random_target_end_index = torch.tensor([sp[2] for sp, dsi in zip(random_target_spans_for_loss, document_start_indices)]).to(device)

                # feed into model
                outputs = model(**inputs, start_positions=random_target_start_index, end_positions=random_target_end_index)

                # calculate loss
                loss = outputs.loss
                loss = loss / args.gradient_accumulation_steps 

                # process model output
                start_logprob = F.log_softmax(outputs.start_logits, dim=-1)
                end_logprob = F.log_softmax(outputs.end_logits, dim=-1)
                f1s, ems, ff1s = [], [], []
                for start_logp, end_logp, tspans, document_start_index, word2time, input_ids\
                        in zip(start_logprob, end_logprob, target_spans, document_start_indices, batch['word2time'], inputs['input_ids']):
                    # calculate predicted spans
                    predicted_start_index, predicted_end_index = post_process_prediction(start_logp[document_start_index+1:], end_logp[document_start_index+1:], 3, 50) 
                    
                    # calculate start and end time of spans
                    # (token -> word -> spans)
                    predicted_start_word_index, predicted_end_word_index, tokens = align_word_index(\
                            input_ids[document_start_index+1:], \
                            tokenizer, \
                            pred_span=[predicted_start_index, predicted_end_index],\
                            document_start_index=document_start_index, \
                            sep_id=tokenizer.sep_token_id, \
                            space_token=space_token)
                    word2time_entities = word2time[predicted_start_word_index:predicted_end_word_index+1] 
                    #import pdb;pdb.set_trace()
                    # get predicted_start_time and predicted_time
                    try:
                        predicted_start_time = float(word2time_entities[0][2])
                        predicted_end_time = float(word2time_entities[-1][3])
                    except:
                        #import pdb;pdb.set_trace()
                        #print(word2time_entities)
                        #print(done)
                        continue

                    # calculate F1 and EM score
                    max_f1 = -1
                    max_em = -1
                    max_ff1 = -1
                    for _, target_start_index, target_end_index, target_start_time, target_end_time in tspans:
                        target_start_index -= document_start_index + 1
                        target_end_index -= document_start_index + 1

                        # let f1 and em =0
                        #f1, em = F1EM_score(predicted_start_index, predicted_end_index+1, target_start_index, target_end_index+1)
                        f1, em = 0, 0
                        ff1 = Frame_F1_score(predicted_start_time, predicted_end_time, target_start_time, target_end_time)
                        max_f1, max_em = max([max_f1, max_em], [f1, em], key=lambda x:x[0])
                        max_ff1 = max(max_ff1, ff1)
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
                #tbar.set_postfix({'loss':round(ave_loss, 2), 'F1':round(F1, 2)})
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
    writer.add_scalar('F1/test', ave_F1, n)
    writer.add_scalar('EM/test', ave_EM, n)
    writer.add_scalar('FF1/test', ave_FF1, n)
    writer.add_scalar('Loss/test', ave_Loss, n)


    # save last model
    os.makedirs(ckpt_dir, exist_ok=True)
    last_model_path = os.path.join(ckpt_dir, f'last_epoch{n}_FF1-{ave_FF1:.2f}.ckpt')
    torch.save(model.state_dict(), last_model_path)
    if n>0:
        os.remove(former_last_model_path)
    former_last_model_path = last_model_path
    # save best model
    if ave_FF1 > best_FF1:
        print(f'save best model, epoch={n}, FF1={ave_FF1:.2f}')
        best_model_path = os.path.join(ckpt_dir, f'best_epoch{n}_FF1-{ave_FF1:.2f}.ckpt')
        torch.save(model.state_dict(), best_model_path)
        if n>0:
            os.remove(former_best_model_path)
        former_best_model_path = best_model_path
        best_FF1 = ave_FF1

