


dataset_name=movies
dataset_root=/root/codespace/xai_benchmark/benchmark_data/movies
models=( "bert-base-uncased" "distilbert-base-uncased" "ernie-2.0-base-en" "roberta-base" )
# models=( "bert-base-uncased" )

gpu=4
max_seq_len=512
for n in "${!models[@]}"; do
    model=${models[n]}
    echo $model
    CUDA_VISIBLE_DEVICES=$((gpu)) nohup python evaluate_2.py --dataset_name $dataset_name --dataset_root $dataset_root --eval_configs "{\"max_seq_len\": ${max_seq_len}, \"is_random_samples\": 1}" --model_name $model >> output/eval2__${model}_${dataset_name}.log &
    pid[$n]=$!
    echo $gpu start: pid=$!
    gpu=$((gpu+1))
    if [[ $gpu -eq 8 ]]; then gpu=0; echo "8 tasks are running."; wait; fi
    # if [[ $gpu -eq 8 ]]; then gpu=1; echo "8 tasks are running."; wait; fi
done

models=( "bert-large-uncased" )
gpu=5
max_seq_len=300
for n in "${!models[@]}"; do
    model=${models[n]}
    echo $model
    CUDA_VISIBLE_DEVICES=$((gpu)) nohup python evaluate_2.py --dataset_name $dataset_name --dataset_root $dataset_root --eval_configs "{\"max_seq_len\": ${max_seq_len}, \"is_random_samples\": 1}" --model_name $model >> output/eval2__${model}_${dataset_name}.log &
    pid[$n]=$!
    echo $gpu start: pid=$!
    gpu=$((gpu+1))
    if [[ $gpu -eq 8 ]]; then gpu=0; echo "8 tasks are running."; wait; fi
    # if [[ $gpu -eq 8 ]]; then gpu=1; echo "8 tasks are running."; wait; fi
done
