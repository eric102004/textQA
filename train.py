import math
import time
import random
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import DebertaTokenizer, DebertaForQuestionAnswering
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard.writer import SummaryWriter

from dataset import TextQADataset, collate_fn
from scheduler import update_lr

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
args = parser.parse_args()

# make log_dir and ckpt_dir
subdir = time.strftime("%Y%m%d_%H%M%S", time.localtime())
print(f'save checkpoint and log at subdir: {subdir}')
ckpt_dir = os.path.join(args.ckpt_dir, subdir)
log_dir = os.path.join(args.log_dir, subdir)
os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# fix random seed
from pytorch_lightning import seed_everything
seed_everything(42)


# init dataset and dataloader (train and dev set)
train_dataset = TextQADataset(args.train_meta_path)
test_dataset = TextQADataset(args.test_meta_path)
train_dataloder = DataLoader(training_dataset, batch_size=args.train_batch_size, shuffle=True, num_worker=8)
test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_worker=8)

# calculate total steps
total_steps = math.ceil(len(train_dataloader)/args.gradient_accumulation_steps)*args.n_epoch

# init model and optimizer
model = DebertaForQuestionAnswering.from_pretrained(args.LM_path)
raise NotImplementedError('must check grad of each model parameters')
optimizer = Adam(model.parameters(), lr=args.lr, eps=1e-6, weight_decay=0)
epsilon = 1e-6
beta1 = 0.9
beta2 = 0.999
gradient_clipping = 1.0
lr_decay = linear
weight_decay = None


# load pretrined weights
#model.deberta.load_state_dict(torch.load(args.LM_path))

# init tensorboard
writer = SummaryWriter(log_dir=log_dir)





# start training
best_F1 = -1
cur_step = 1
for n in range(n_epoch):
    print(f'epoch:{n}')
    print('Training')
    model.train()
    F1s = []
    EMs = []
    losses = []
    with tqdm(train_dataloader, unit='batch') as tbar:
        for batch_idx, batch in enumerate(tbar):

            # preprocess batch
            inputs = batch['input ids'].to(device)
            document_start_index = batch['document_start_index']
            target_spans = batch['token answer spans']
            random_target_spans_for_loss = [random.choise(sps) for sps in target_spans] 
            random_target_start_index = torch.tensor([sp[0] for sp in random_target_spans_for_loss]).to(device)
            random_target_end_index = torch.tensor([sp[1] for sp in random_target_spans_for_loss]).to(device)
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
            for start_logp, end_logp in zip(start_logprob, end_logprob):
                # calculate predicted spans
                predicted_start_index, predicted_end_index = post_process_preidction(start_logp, end_logp, 3, 50) 
                # calculate F1 and EM score
                max_f1 = -1
                max_em = -1
                for target_start_index, target_end_index in target_spans:
                    target_start_index -= document_start_index + 1
                    target_end_index -= document_start_index + 1

                    _, _, f1, em = F1EM_score(predicted_start_index, predicted_end_inex+1, target_start_index, target_end_index+1)
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
                update_lr(optimizer, args.lr, cur_step, total_steps)
                optimizer.step()
                cur_step += 1
                
            # record output and scores
            tbar.set_postfix({'loss':round(ave_loss, 2), 'F1':round(F1, 2)})
            raise NotImplementedError('must add tensorboard')

    # calcualte average of scores
    ave_F1 = sum(F1) / len(F1)
    ave_EM = sum(EM) / len(EM)
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
    losses = []
    with torch.no_grad():
        with tqdm(test_dataloader, unit='batch') as tbar:
            for batch_idx, batch in enumerate(tbar):
                # preprocess batch
                inputs = batch['input ids'].to(device)
                document_start_index = batch['document_start_index']
                target_spans = batch['token answer spans']
                random_target_spans_for_loss = [random.choise(sps) for sps in target_spans] 
                random_target_start_index = torch.tensor([sp[0] for sp in random_target_spans_for_loss]).to(device)
                random_target_end_index = torch.tensor([sp[1] for sp in random_target_spans_for_loss]).to(device)
                # feed into model
                outputs = model(**inputs, start_positions=random_target_start_index, end_positions=random_target_end_index)

                # calculate loss
                loss = outputs.loss
                loss = loss / args.gradient_accumulation_steps 

                # process model output
                start_logprob = F.log_softmax(outputs.start_logits, dim=-1)
                end_logprob = F.log_softmax(outputs.end_logits, dim=-1)
                f1s, ems = [], []
                for start_logp, end_logp in zip(start_logprob, end_logprob):
                    # calculate predicted spans
                    predicted_start_index, predicted_end_index = post_process_preidction(start_logp, end_logp, 3, 50) 
                    # calculate F1 and EM score
                    max_f1 = -1
                    max_em = -1
                    for target_start_index, target_end_index in target_spans:
                        target_start_index -= document_start_index + 1
                        target_end_index -= document_start_index + 1

                        _, _, f1, em = F1EM_score(predicted_start_index, predicted_end_inex+1, target_start_index, target_end_index+1)
                        max_f1, max_em = max([max_f1, max_em], [f1, em], key=lambda x:x[0])
                    f1s.append(max_f1)
                    ems.append(max_em)
                F1 = sum(f1s) / len(f1s)
                EM = sum(ems) / len(ems)
                ave_loss = loss.mean().items()
                F1s.append(F1)
                EMs.append(EM)
                losses.append(ave_loss)

                # record output and scores
                tbar.set_postfix({'loss':round(loss, 2), 'F1':round(F1, 2)})

    # calcualte average of scores
    ave_F1 = sum(F1) / len(F1)
    ave_EM = sum(EM) / len(EM)
    ave_Loss = sum(losses) / len(losses)
    print(f'Average F1: {ave_F1}')
    print(f'Average EM: {ave_EM}')
    print(f'Average loss: {ave_Loss}')
    writer.add_scalar('F1/test', ave_F1, n)
    writer.add_scalar('EM/test', ave_EM, n)
    writer.add_scalar('Loss/test', ave_Loss, n)


    # save last model
    last_model_path = os.path.join(ckpt_dir, f'last_epoch{n}_F1-{ave_F1:.2f}.ckpt')
    torch.save(model.state_dict(), last_model_path)
    if n>0:
        os.remove(former_last_model_path)
    former_last_model_path = last_model_path
    # save best model
    if ave_F1 > best_F1:
        best_model_path = os.path.join(ckpt_dir, f'best_epoch{n}_F1-{ave_F1:.2f}.ckpt')
        torch.save(model.state_dict(), best_model_path)
        if n>0:
            os.remove(former_best_model_path)
        former_best_model_path = best_model_path
        best_F1 = ave_F1


