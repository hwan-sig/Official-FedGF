#!/usr/bin/env bash
cd ../

declare -a algorithms=('fedgf')
#'fedavg' 'fedavgm' 'feddyn' 'fedprox' 'scaffold'
# 'fedsam' 'fedasam' 'mofedsam'
# 'fedsmoo_noreg' 'fedgf'
declare -a datasets=('cifar10' 'cifar100')
declare -a models=('FedSAMcnn')
declare wandb_project_name="cifar"
declare -a sample_ratios=('0.05' '0.1' '0.2')

declare batch_sizes=('64')
declare epochs=('1')
declare lrs=('0.01')
declare wd=0.0004

declare gpu_start=0
declare gpu_end=8
((gpu=gpu_start))

declare num_cpu=9

rhos=('0.02' '0.05' '0.1')
c_os=('0.2' '0.3')

function set_param() {
    if [ "${dataset}" = "cifar10" ]; then
      dir_alphas=("0" "0.05" "100")
      eval_every=800
      round=10000
    elif [ "${dataset}" = "cifar100" ]; then
      dir_alphas=("0" "0.5" "1000")
      eval_every=1000
      round=20000
    fi
}

function set_gpu() {
  gpu=$((gpu + 1))
  if [ $gpu -gt $gpu_end ]; then
    gpu=$gpu_start
  fi
}

function run_fedavg() {
  CUDA_VISIBLE_DEVICES=$((gpu)) taskset --cpu-list $((gpu*num_cpu))-$((gpu*num_cpu+num_cpu-1)) \
  nohup python -u main.py --eval_every $eval_every --wandb_project_name $wandb_project_name  --model $model --alg $algorithm --dataset $dataset --total_client 100 \
  --com_round $round --sample_ratio $sample_ratio --batch_size $batch_size --epochs $epoch --lr $lr --weight_decay $wd \
  --dir_alpha $dir_alpha --transform --save_model &

  set_gpu
}
function run_fedavgm() {
  for beta in "${betas[@]}"; do
    CUDA_VISIBLE_DEVICES=$((gpu)) taskset --cpu-list $((gpu*num_cpu))-$((gpu*num_cpu+num_cpu-1)) \
    nohup python -u main.py --eval_every $eval_every --wandb_project_name $wandb_project_name --model $model --alg $algorithm --dataset $dataset --total_client 100 \
    --com_round $round --sample_ratio $sample_ratio --batch_size $batch_size --epochs $epoch --lr $lr --weight_decay $wd --beta $beta \
    --dir_alpha $dir_alpha --transform --save_model &

    set_gpu
  done
}

function run_scaffold() {
  g_lrs=('0.01' '0.001')

  for g_lr in "${g_lrs[@]}"; do
    CUDA_VISIBLE_DEVICES=$((gpu)) taskset --cpu-list $((gpu*num_cpu))-$((gpu*num_cpu+num_cpu-1)) \
    nohup python -u main.py --eval_every $eval_every --wandb_project_name $wandb_project_name --model $model --alg $algorithm --dataset $dataset --total_client 100 \
    --com_round $round --sample_ratio $sample_ratio --batch_size $batch_size --epochs $epoch --lr $lr --weight_decay $wd --g_lr $g_lr \
    --dir_alpha $dir_alpha --transform --save_model &

    set_gpu
  done
}

function run_fedprox() {
  mus=('0.01' '0.001')

  for mu in "${mus[@]}"; do
    CUDA_VISIBLE_DEVICES=$((gpu)) taskset --cpu-list $((gpu*num_cpu))-$((gpu*num_cpu+num_cpu-1)) \
    nohup python -u main.py --eval_every $eval_every --wandb_project_name $wandb_project_name --model $model --alg $algorithm --dataset $dataset --total_client 100 \
    --com_round $round --sample_ratio $sample_ratio --batch_size $batch_size --epochs $epoch --lr $lr --weight_decay $wd --mu $mu \
    --dir_alpha $dir_alpha --transform--save_model &

    set_gpu
  done
}

function run_feddyn() {
  alphas=('0.1' '0.01' '0.001')

  for alpha in "${alphas[@]}"; do
    CUDA_VISIBLE_DEVICES=$((gpu)) taskset --cpu-list $((gpu*num_cpu))-$((gpu*num_cpu+num_cpu-1)) \
    nohup python -u main.py --eval_every $eval_every --wandb_project_name $wandb_project_name --model $model --alg $algorithm --dataset $dataset --total_client 100 \
    --com_round $round --sample_ratio $sample_ratio --batch_size $batch_size --epochs $epoch --lr $lr --weight_decay $wd --alpha $alpha \
    --dir_alpha $dir_alpha --transform--save_model &

    set_gpu
  done
}

function run_fedsam() {
  if [ "${dataset}" = "cifar10" ]; then
    if [ "${dir_alpha}" = "0" ]; then
      rho=0.1
    elif [ "${dir_alpha}" = "0.05" ]; then
      rho=0.1
    elif [ "${dir_alpha}" = "100" ]; then
      rho=0.02
    fi
  elif [ "${dataset}" = "cifar100" ]; then
    if [ "${dir_alpha}" = "0" ]; then
      rho=0.02
    elif [ "${dir_alpha}" = "0.5" ]; then
      rho=0.05
    elif [ "${dir_alpha}" = "1000" ]; then
      rho=0.05
    fi
  fi

  CUDA_VISIBLE_DEVICES=$((gpu)) taskset --cpu-list $((gpu*num_cpu))-$((gpu*num_cpu+num_cpu-1)) \
  nohup python -u main.py --eval_every $eval_every --wandb_project_name $wandb_project_name --model $model --alg $algorithm --dataset $dataset --total_client 100 \
  --com_round $round --sample_ratio $sample_ratio --batch_size $batch_size --epochs $epoch --lr $lr --weight_decay $wd --rho $rho \
  --dir_alpha $dir_alpha --transform--save_model &
  set_gpu
}

function run_fedasam() {
  if [ "${dataset}" = "cifar10" ]; then
    if [ "${dir_alpha}" = "0" ]; then
      rho=0.7
      eta=0.2
    elif [ "${dir_alpha}" = "0.05" ]; then
      rho=0.7
      eta=0.2
    elif [ "${dir_alpha}" = "100" ]; then
      rho=0.05
      eta=0.2
    fi
  elif [ "${dataset}" = "cifar100" ]; then
    if [ "${dir_alpha}" = "0" ]; then
      rho=0.5
      eta=0.2
    elif [ "${dir_alpha}" = "0.5" ]; then
      rho=0.5
      eta=0.2
    elif [ "${dir_alpha}" = "1000" ]; then
      rho=0.5
      eta=0.2
    fi
  fi

  CUDA_VISIBLE_DEVICES=$((gpu)) taskset --cpu-list $((gpu*num_cpu))-$((gpu*num_cpu+num_cpu-1)) \
  nohup python -u main.py --eval_every $eval_every --wandb_project_name $wandb_project_name --model $model --alg $algorithm --dataset $dataset --total_client 100 \
  --com_round $round --sample_ratio $sample_ratio --batch_size $batch_size --epochs $epoch --lr $lr --weight_decay $wd --rho $rho --eta $eta \
  --dir_alpha $dir_alpha --transform --save_model &

  set_gpu
}

function run_mofedsam() {
  betas=("0.1" "0.7" "0.9")

  for rho in "${rhos[@]}"; do
    for beta in "${betas[@]}"; do
      CUDA_VISIBLE_DEVICES=$((gpu)) taskset --cpu-list $((gpu*num_cpu))-$((gpu*num_cpu+num_cpu-1)) \
      nohup python -u main.py --eval_every $eval_every --wandb_project_name $wandb_project_name --model $model --alg $algorithm --dataset $dataset --total_client 100 \
      --com_round $round --sample_ratio $sample_ratio --batch_size $batch_size --epochs $epoch --lr $lr --weight_decay $wd --rho $rho --beta $beta \
      --dir_alpha $dir_alpha --transform --save_model &

      set_gpu
    done
  done
}

function run_fedgf() {
  W="30"

  for rho in "${rhos[@]}"; do
    for c_o in "${c_os[@]}"; do
      CUDA_VISIBLE_DEVICES=$((gpu)) taskset --cpu-list $((gpu*num_cpu))-$((gpu*num_cpu+num_cpu-1)) \
      nohup python -u main.py --eval_every $eval_every --wandb_project_name $wandb_project_name --model $model --alg $algorithm --dataset $dataset --total_client 100 \
      --com_round $round --sample_ratio $sample_ratio --batch_size $batch_size --epochs $epoch --lr $lr --weight_decay $wd --rho $rho \
      --c_o $c_o --dir_alpha $dir_alpha --transform --W $W --save_model &
      set_gpu
    done
  done
}

function run_fedsmoo_noreg() {
  for rho in "${rhos[@]}"; do
    CUDA_VISIBLE_DEVICES=$((gpu)) taskset --cpu-list $((gpu*num_cpu))-$((gpu*num_cpu+num_cpu-1)) \
    nohup python -u main.py --eval_every $eval_every --wandb_project_name $wandb_project_name --model $model --alg $algorithm --dataset $dataset --total_client 100 \
    --com_round $round --sample_ratio $sample_ratio --batch_size $batch_size --epochs $epoch --lr $lr --weight_decay $wd --rho $rho \
    --dir_alpha $dir_alpha --transform --save_model &

    set_gpu
  done
}

function run_fedgamma() {
  for rho in "${rhos[@]}"; do
    CUDA_VISIBLE_DEVICES=$((gpu)) taskset --cpu-list $((gpu*num_cpu))-$((gpu*num_cpu+num_cpu-1)) \
    nohup python -u main.py --eval_every $eval_every --wandb_project_name $wandb_project_name --model $model --alg $algorithm --dataset $dataset --total_client 100 \
    --com_round $round --sample_ratio $sample_ratio --batch_size $batch_size --epochs $epoch --lr $lr --weight_decay $wd --rho $rho \
    --dir_alpha $dir_alpha --transform --save_model &

    set_gpu
  done
}

for dataset in "${datasets[@]}"; do
  echo "############################################## Running ##############################################"
  set_param
  for dir_alpha in "${dir_alphas[@]}"; do
    for model in "${models[@]}"; do
      for lr in "${lrs[@]}"; do
        for batch_size in "${batch_sizes[@]}"; do
          for epoch in "${epochs[@]}"; do
            for algorithm in "${algorithms[@]}"; do
              for sample_ratio in "${sample_ratios[@]}"; do
                echo $gpu

                if [ "${algorithm}" = "fedavg" ]; then
                  run_fedavg
                elif [ "${algorithm}" = "feddyn" ]; then
                  run_feddyn
                elif [ "${algorithm}" = "fedavgm" ]; then
                  run_fedavgm
                elif [ "${algorithm}" = "scaffold" ]; then
                  run_scaffold
                elif [ "${algorithm}" = "fedprox" ]; then
                  run_fedprox
                elif [ "${algorithm}" = "fedsam" ]; then
                  run_fedsam
                elif [ "${algorithm}" = "fedasam" ]; then
                  run_fedasam
                elif [ "${algorithm}" = "mofedsam" ]; then
                  run_mofedsam
                elif [ "${algorithm}" = "fedgf" ]; then
                  run_fedgf
                elif [ "${algorithm}" = "fedsmoo_noreg" ]; then
                  run_fedsmoo_noreg
                elif [ "${algorithm}" = "fedgamma" ]; then
                  run_fedgamma
                fi
              done
            done
          done
        done
      done
    done
  done
done
