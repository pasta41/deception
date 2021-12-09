cd ..

base_dir=`pwd`

MODEL=logistic_regression
DATASET=MNIST
OPTIM=SGD
SEED=1

JOB_NAME=${MODEL}_${DATASET}_${OPTIM}_${SEED}


run_cmd="python3 train.py --dir=${base_dir}/ckpts \
    --job_type=vision \
    --job_name=${JOB_NAME} \
    --dataset=${DATASET} \
    --data_path=${base_dir}/data \
    --batch_size=128 \
    --model=${MODEL} \
    --optim=${OPTIM} \
    --epochs=100 \
    --lr=0.1 \
    --momentum=0.9 \
    --wd=5e-4 \
    --seed=1 \
    --use_tensorboard \
    --tensorboard_dir=${base_dir}/runs/${JOB_NAME} \
    --use_cuda \
    --use_data_parallel"

echo ${run_cmd}
eval ${run_cmd}