TRAIN_META_PATH=../slue-sqa5/slue-sqa5_fine-tune.tsv
TEST_META_PATH=../slue-sqa5/slue-sqa5_dev.tsv

LM_PATH=microsoft/deberta-base

CKPT_DIR=ckpt
LOG_DIR=log

TEXT_SOURCE=groundtruth
MODE=train

python3 train.py \
    --train-meta-path $TRAIN_META_PATH \
    --test-meta-path $TEST_META_PATH \
    --LM-path $LM_PATH \
    --ckpt-dir $CKPT_DIR \
    --log-dir $LOG_DIR \
    --lr 4e-5 \
    --max-grad-norm 1.0 \
    --warmup-steps 50 \
    --n-epoch 10 \
    --train-batch-size 16 \
    --test-batch-size 16 \
    --gradient-accumulation-steps 4 \
    --source $TEXT_SOURCE \
    --mode $MODE \

