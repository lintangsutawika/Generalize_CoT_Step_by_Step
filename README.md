# Generalizing CoT with Filler Tokens

This repository is based on code from the paper [From Explicit CoT to Implicit CoT: Learning to Internalize CoT Step by Step](https://arxiv.org/pdf/2405.14838). It's original version can be found [here](https://github.com/da03/Internalize_CoT_Step_by_Step).

### Training

To train the model, run the following commands. The example uses 9 X 9 Mult with GPT-2:

```
export D=4
export DATANAME=lintang/implicit-cot-math
export DATAPATH=${D}_by_${D}_mult
export MODEL=gpt2
export EPOCHS=200
export LR=5e-5
export BSZ=32
export ACCUMULATE=1
export REMOVE_PER_EPOCH=8
export REMOVE_ALL_WHEN_REMOVE_BEYOND=inf
export REMOVAL_SMOOTHING_LAMBDA=4
export REMOVAL_SIDE=left
export PRETRAIN_EPOCHS=0
export SEED=3456
export SAVE=/data/user_data/lsutawik/rpt/train_models/${D}_by_${D}_mult/gpt2
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python src/train.py \
    --model ${MODEL} \
    --data_path ${DATANAME} \
    --data_name ${DATAPATH} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --batch_size ${BSZ} \
    --accumulate ${ACCUMULATE} \
    --remove_per_epoch ${REMOVE_PER_EPOCH} \
    --remove_all_when_remove_beyond ${REMOVE_ALL_WHEN_REMOVE_BEYOND} \
    --removal_smoothing_lambda ${REMOVAL_SMOOTHING_LAMBDA} \
    --removal_side ${REMOVAL_SIDE} \
    --pretrain_epochs ${PRETRAIN_EPOCHS} \
    --seed ${SEED} \
    --reset_optimizer \
    --save_model ${SAVE} \
    > ${SAVE}/log.train 2>&1
```

### Generation & Evaluation

Here we use a pretrained model as an example. Download the folder `models/9_by_9_mult/gpt2`, then the following command will run inference and evaluate both accuracy and throughput, logged in file `generation_logs/9_by_9_mult/log.generate`.

```
export D=9
export FOLDER=data/${D}_by_${D}_mult/
export MODEL=models/${D}_by_${D}_mult/gpt2
export BSZ=1
export SAVE=generation_logs/${D}_by_${D}_mult/gpt2
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python src/generate.py \
    --from_pretrained ${MODEL} \
    --test_path ${FOLDER}/test_bigbench.txt \
    --batch_size ${BSZ} \
    > ${SAVE}/log.generate 2>&1&
```

### Command for GSM8K

```
export FOLDER=data/gsm8k
export MODEL=mistralai/Mistral-7B-v0.1
export EPOCHS=80
export LR=1e-5
export BSZ=16
export ACCUMULATE=2
export REMOVE_PER_EPOCH=8
export REMOVE_ALL_WHEN_REMOVE_BEYOND=39
export MAX_LEN_TRAIN=150
export REMOVAL_SMOOTHING_LAMBDA=4
export REMOVAL_SIDE=left
export PRETRAIN_EPOCHS=0
export SEED=1234
export SAVE=train_models/gsm8k
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python src/train.py \
    --model ${MODEL} \
    --train_path ${FOLDER}/train.txt \
    --val_path ${FOLDER}/valid.txt \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --batch_size ${BSZ} \
    --accumulate ${ACCUMULATE} \
    --remove_per_epoch ${REMOVE_PER_EPOCH} \
    --remove_all_when_remove_beyond ${REMOVE_ALL_WHEN_REMOVE_BEYOND} \
    --removal_smoothing_lambda ${REMOVAL_SMOOTHING_LAMBDA} \
    --removal_side ${REMOVAL_SIDE} \
    --pretrain_epochs ${PRETRAIN_EPOCHS} \
    --seed ${SEED} \
    --reset_optimizer \
    --bf16 \
    --max_len_train ${MAX_LEN_TRAIN} \
    --save_model ${SAVE} \
    > ${SAVE}/log.train 2>&1
```
