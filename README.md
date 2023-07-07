# textQA
Training/Evaluation scripts for pipeline baseline in SLUE QA benchmark

## Fine-tuning
Fine-tune the text QA model with the groundtruth transcripts of the SLUE-SQA-5 fine-tune set
```
 sh train.sh
```

## Evaluation
Evaluate the text QA model with the groundtruth transcripts of the SLUE-SQA-5 dev set
```
sh eval_groundtruth.sh
```
