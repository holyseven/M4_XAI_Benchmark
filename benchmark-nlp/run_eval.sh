

dataset_name=movies
dataset_root=/root/codespace/xai_benchmark/benchmark_data/movies

rs_path=( "default-logit__bert-base-uncased__bt-head-0__movies.npz" "default-logit__bert-base-uncased__bt-head-10__movies.npz" "default-logit__bert-base-uncased__bt-head-11__movies.npz" "default-logit__bert-base-uncased__bt-head-2__movies.npz" "default-logit__bert-base-uncased__bt-head-4__movies.npz" "default-logit__bert-base-uncased__bt-head-6__movies.npz" "default-logit__bert-base-uncased__bt-head-8__movies.npz" "default-logit__bert-base-uncased__bt-token-0__movies.npz" "default-logit__bert-base-uncased__bt-token-10__movies.npz" "default-logit__bert-base-uncased__bt-token-11__movies.npz" "default-logit__bert-base-uncased__bt-token-2__movies.npz" "default-logit__bert-base-uncased__bt-token-4__movies.npz" "default-logit__bert-base-uncased__bt-token-6__movies.npz" "default-logit__bert-base-uncased__bt-token-8__movies.npz" "default-logit__bert-large-uncased__bt-head-0__movies.npz" "default-logit__bert-large-uncased__bt-head-12__movies.npz" "default-logit__bert-large-uncased__bt-head-16__movies.npz" "default-logit__bert-large-uncased__bt-head-20__movies.npz" "default-logit__bert-large-uncased__bt-head-23__movies.npz" "default-logit__bert-large-uncased__bt-head-4__movies.npz" "default-logit__bert-large-uncased__bt-head-8__movies.npz" "default-logit__bert-large-uncased__bt-token-0__movies.npz" "default-logit__bert-large-uncased__bt-token-12__movies.npz" "default-logit__bert-large-uncased__bt-token-16__movies.npz" "default-logit__bert-large-uncased__bt-token-20__movies.npz" "default-logit__bert-large-uncased__bt-token-23__movies.npz" "default-logit__bert-large-uncased__bt-token-4__movies.npz" "default-logit__bert-large-uncased__bt-token-8__movies.npz" "default-logit__distilbert-base-uncased__bt-head-0__movies.npz" "default-logit__distilbert-base-uncased__bt-head-1__movies.npz" "default-logit__distilbert-base-uncased__bt-head-2__movies.npz" "default-logit__distilbert-base-uncased__bt-head-3__movies.npz" "default-logit__distilbert-base-uncased__bt-head-4__movies.npz" "default-logit__distilbert-base-uncased__bt-head-5__movies.npz" "default-logit__distilbert-base-uncased__bt-token-0__movies.npz" "default-logit__distilbert-base-uncased__bt-token-1__movies.npz" "default-logit__distilbert-base-uncased__bt-token-2__movies.npz" "default-logit__distilbert-base-uncased__bt-token-3__movies.npz" "default-logit__distilbert-base-uncased__bt-token-4__movies.npz" "default-logit__distilbert-base-uncased__bt-token-5__movies.npz" "default-logit__ernie-2.0-base-en__bt-head-0__movies.npz" "default-logit__ernie-2.0-base-en__bt-head-10__movies.npz" "default-logit__ernie-2.0-base-en__bt-head-11__movies.npz" "default-logit__ernie-2.0-base-en__bt-head-2__movies.npz" "default-logit__ernie-2.0-base-en__bt-head-4__movies.npz" "default-logit__ernie-2.0-base-en__bt-head-6__movies.npz" "default-logit__ernie-2.0-base-en__bt-head-8__movies.npz" "default-logit__ernie-2.0-base-en__bt-token-0__movies.npz" "default-logit__ernie-2.0-base-en__bt-token-10__movies.npz" "default-logit__ernie-2.0-base-en__bt-token-11__movies.npz" "default-logit__ernie-2.0-base-en__bt-token-2__movies.npz" "default-logit__ernie-2.0-base-en__bt-token-4__movies.npz" "default-logit__ernie-2.0-base-en__bt-token-6__movies.npz" "default-logit__ernie-2.0-base-en__bt-token-8__movies.npz" "default-logit__roberta-base__bt-head-0__movies.npz" "default-logit__roberta-base__bt-head-10__movies.npz" "default-logit__roberta-base__bt-head-11__movies.npz" "default-logit__roberta-base__bt-head-2__movies.npz" "default-logit__roberta-base__bt-head-4__movies.npz" "default-logit__roberta-base__bt-head-6__movies.npz" "default-logit__roberta-base__bt-head-8__movies.npz" "default-logit__roberta-base__bt-token-0__movies.npz" "default-logit__roberta-base__bt-token-10__movies.npz" "default-logit__roberta-base__bt-token-11__movies.npz" "default-logit__roberta-base__bt-token-2__movies.npz" "default-logit__roberta-base__bt-token-4__movies.npz" "default-logit__roberta-base__bt-token-6__movies.npz" "default-logit__roberta-base__bt-token-8__movies.npz" )


gpu=0
for n in "${!rs_path[@]}"; do
    exp_path=${rs_path[n]}
    splits=(${exp_path//__/ })
    model=${splits[1]}
    echo ${rs_path[n]}
    echo $model
    echo $max_seq_len

    max_seq_len=512
    if [[ "$model" == "bert-large-uncased" ]]
    then 
        max_seq_len=300 
    fi
    
    CUDA_VISIBLE_DEVICES=$((gpu)) nohup python evaluate_faithfulness.py --save_eval_result 1 --dataset_name $dataset_name --dataset_root $dataset_root --model_name $model --eval_configs "{\"max_seq_len\": ${max_seq_len}, \"percentile\": 1}" --exp_path work_dirs/$exp_path >> output/eval-l__${exp_path}.log &
    pid[$n]=$!
    echo $gpu start: pid=$!
    gpu=$((gpu+1))
    if [[ $gpu -eq 8 ]]; then gpu=0; echo "8 tasks are running."; wait; fi
done


# rs_path=( "default-wev4__bert-base-uncased__bt-token-2__movies.npz" "default-wev4__bert-base-uncased__bt-token-4__movies.npz" "default-wev4__bert-base-uncased__bt-token-6__movies.npz" "default-wev4__bert-base-uncased__bt-token-8__movies.npz" "default-wev4__bert-base-uncased__bt-token-10__movies.npz" "default-wev4__bert-base-uncased__bt-token-11__movies.npz" "default-wev4__bert-base-uncased__bt-head-2__movies.npz" "default-wev4__bert-base-uncased__bt-head-4__movies.npz" "default-wev4__bert-base-uncased__bt-head-6__movies.npz" "default-wev4__bert-base-uncased__bt-head-8__movies.npz" "default-wev4__bert-base-uncased__bt-head-10__movies.npz" "default-wev4__bert-base-uncased__bt-head-11__movies.npz" )

# rs_path=( "default-logit__bert-base-uncased__intgrad__movies.npz" "default-logit__bert-base-uncased__smoothgrad__movies.npz" "default-logit__bert-base-uncased__ga-0__movies.npz" "default-logit__bert-base-uncased__ga-2__movies.npz" "default-logit__bert-base-uncased__ga-4__movies.npz" "default-logit__bert-base-uncased__ga-6__movies.npz" "default-logit__bert-base-uncased__ga-8__movies.npz" "default-logit__bert-base-uncased__ga-10__movies.npz" "default-logit__bert-base-uncased__ga-11__movies.npz" )

# gpu=0
# for n in "${!rs_path[@]}"; do
#     echo ${rs_path[n]}
#     exp_path=${rs_path[n]}
#     model="bert-base-uncased"

#     max_seq_len=512
#     if [[ "$model" == "bert-large-uncased" ]]
#     then 
#         max_seq_len=300 
#     fi

#     CUDA_VISIBLE_DEVICES=$((gpu)) nohup python evaluate_faithfulness.py --save_eval_result 1 --dataset_name $dataset_name --dataset_root $dataset_root --model_name $model --eval_configs "{\"max_seq_len\": ${max_seq_len}, \"percentile\": 1}" --exp_path work_dirs/$exp_path >> output/eval-l__${exp_path}.log &
#     pid[$n]=$!
#     echo $gpu start: pid=$!
#     gpu=$((gpu+1))
#     if [[ $gpu -eq 8 ]]; then gpu=0; echo "8 tasks are running."; wait; fi
# done
