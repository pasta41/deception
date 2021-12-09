cd ..

base_dir=`pwd`

MODEL=LSTM
DATASET=wikitext-2
OPTIM=SGD
SEED=1

JOB_NAME=${MODEL}_${DATASET}_${OPTIM}_${SEED}


run_cmd="python3 train.py --dir=${base_dir}/ckpts \
    --job_type=nlp \
    --job_name=${JOB_NAME} \
    --dataset=${DATASET} \
    --data_path=${base_dir}/data \
    --batch_size=20 \
    --model=${MODEL} \
    --optim=${OPTIM} \
    --epochs=50 \
    --lr=20 \
    --momentum=0.9 \
    --seed=1 \
    --use_tensorboard \
    --tensorboard_dir=${base_dir}/runs/${JOB_NAME} \
    --use_cuda \
    --use_data_parallel \
    --emsize=200 \
    --nhid=200 \
    --nlayers=2 \
    --clip=0.25 \
    --bptt=35 \
    --dropout=0.2 \
    --tied \
    --seed=1 \
    --nhead=2"

echo ${run_cmd}
eval ${run_cmd}