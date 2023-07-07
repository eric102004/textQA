# textQA
Training/Evaluation scripts for pipeline baseline in SLUE QA benchmark proposed in the paper [SLUE Phase-2: A Benchmark Suite of Diverse Spoken Language Understanding Tasks](https://arxiv.org/abs/2212.10525)

*** Note: these instructions are a work in progress and will be updated over the next few days

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


