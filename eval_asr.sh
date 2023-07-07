TEST_META_PATH=../slue-sqa5/slue-sqa5_dev.tsv

LM_PATH=microsoft/deberta-base

CKPT_PATH=ckpt/20230517_155149/best_epoch7_FF1-0.74.ckpt


ASR_list=("groundtruth")

for ASR in "${ASR_list[@]}"
do
	echo $ASR
	python3 eval.py \
	    --test-meta-path $TEST_META_PATH \
	    --LM-path $LM_PATH \
	    --ckpt-path $CKPT_PATH \
	    --test-batch-size 16 \
	    --source $ASR \
done
