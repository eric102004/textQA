TRAIN_META_PATH=../QAcheck/processed_filtered_train_top1000_meta.csv
TEST_META_PATH=../QAcheck/processed_filtered_dev_top37374_meta.csv

LM_PATH=

CKPT_DIR=ckpt
LOG_DIR=log

python3 train.py \
    --train-meta-path $TRAIN_META_PATH \
    --test-meta-path $TEST_META_PATH \
    --LM-path $LM_PATH \
    --ckpt-dir $CKPT_DIR \
    --log-dir $LOG_DIR \
    --lr \
    --max-grad-norm 1.0 \
    --warmup-steps \
    --n-epoch \
    --train-batch-size \
    --test-batch-size \
    --gradient-accumulation-steps \




#python3 eval.py \
