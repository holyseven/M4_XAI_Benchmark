


############ smoothgrad #########
dataset_name=movies
dataset_root=/root/codespace/xai_benchmark/benchmark_data/movies
models=( "ernie-2.0-base-en" "bert-base-uncased" "distilbert-base-uncased" "roberta-base" )
gpu=0
# models=( "ernie-2.0-base-en" "bert-base-uncased" "bert-large-uncased" "distilbert-base-uncased" "roberta-base" )


models=( "ernie-2.0-base-en" "bert-base-uncased" "roberta-base" )  # 12 layers
for n in "${!models[@]}"; do
    echo $n ${models[n]}
    
    model=${models[n]}
    layers=( 0 2 4 6 8 10 11 )
    for m in "${!layers[@]}"; do
        sl=${layers[m]}
        # GA
        it="ga"
        CUDA_VISIBLE_DEVICES=${gpu} nohup python interpret_and_save.py --dataset_name $dataset_name --dataset_root $dataset_root --model_name $model --it $it --save_eval_result 1 --it_configs "{\"start_layer\": ${sl}, \"max_seq_len\": 512}" >> output/${model}-${it}-${sl}.log &
        pid[$gpu]=$!
        # manually and naively do the job scheduling for local GPUs.
        echo $gpu start: pid=$!
        gpu=$((gpu+1))
        if [[ $gpu -eq 8 ]]; then gpu=0; echo "8 tasks are running."; wait; fi

        # BT-head
        it="bt"
        CUDA_VISIBLE_DEVICES=${gpu} nohup python interpret_and_save.py --dataset_name $dataset_name --dataset_root $dataset_root --model_name $model --it $it --save_eval_result 1 --it_configs "{\"start_layer\": ${sl}, \"ap_mode\": \"head\", \"max_seq_len\": 512}" >> output/${model}-${it}-head-${sl}.log &
        pid[$gpu]=$!
        # manually and naively do the job scheduling for local GPUs.
        echo $gpu start: pid=$!
        gpu=$((gpu+1))
        if [[ $gpu -eq 8 ]]; then gpu=0; echo "8 tasks are running."; wait; fi

        # BT-token
        it="bt"
        CUDA_VISIBLE_DEVICES=${gpu} nohup python interpret_and_save.py --dataset_name $dataset_name --dataset_root $dataset_root --model_name $model --it $it --save_eval_result 1 --it_configs "{\"start_layer\": ${sl}, \"ap_mode\": \"token\", \"max_seq_len\": 512}" >> output/${model}-${it}-token-${sl}.log &
        pid[$gpu]=$!
        # manually and naively do the job scheduling for local GPUs.
        echo $gpu start: pid=$!
        gpu=$((gpu+1))
        if [[ $gpu -eq 8 ]]; then gpu=0; echo "8 tasks are running."; wait; fi
    done        
done




models=( "distilbert-base-uncased" )  # 6 layers
for n in "${!models[@]}"; do
    echo $n ${models[n]}
    
    model=${models[n]}
    layers=( 0 1 2 3 4 5 )
    for m in "${!layers[@]}"; do
        sl=${layers[m]}
        # GA
        it="ga"
        CUDA_VISIBLE_DEVICES=${gpu} nohup python interpret_and_save.py --dataset_name $dataset_name --dataset_root $dataset_root --model_name $model --it $it --save_eval_result 1 --it_configs "{\"start_layer\": ${sl}, \"max_seq_len\": 512}" >> output/${model}-${it}-${sl}.log &
        pid[$gpu]=$!
        echo $gpu start: pid=$!
        gpu=$((gpu+1))
        if [[ $gpu -eq 8 ]]; then gpu=0; echo "8 tasks are running."; wait; fi

        # BT-head
        it="bt"
        CUDA_VISIBLE_DEVICES=${gpu} nohup python interpret_and_save.py --dataset_name $dataset_name --dataset_root $dataset_root --model_name $model --it $it --save_eval_result 1 --it_configs "{\"start_layer\": ${sl}, \"ap_mode\": \"head\", \"max_seq_len\": 512}" >> output/${model}-${it}-head-${sl}.log &
        pid[$gpu]=$!
        echo $gpu start: pid=$!
        gpu=$((gpu+1))
        if [[ $gpu -eq 8 ]]; then gpu=0; echo "8 tasks are running."; wait; fi

        # BT-token
        it="bt"
        CUDA_VISIBLE_DEVICES=${gpu} nohup python interpret_and_save.py --dataset_name $dataset_name --dataset_root $dataset_root --model_name $model --it $it --save_eval_result 1 --it_configs "{\"start_layer\": ${sl}, \"ap_mode\": \"token\", \"max_seq_len\": 512}" >> output/${model}-${it}-token-${sl}.log &
        pid[$gpu]=$!
        echo $gpu start: pid=$!
        gpu=$((gpu+1))
        if [[ $gpu -eq 8 ]]; then gpu=0; echo "8 tasks are running."; wait; fi
    done
done

wait




############ smoothgrad #########
dataset_name=movies
dataset_root=/root/codespace/xai_benchmark/benchmark_data/movies
models=( "ernie-2.0-base-en" "bert-base-uncased" "distilbert-base-uncased" "roberta-base" )
gpu=0
# models=( "ernie-2.0-base-en" "bert-base-uncased" "bert-large-uncased" "distilbert-base-uncased" "roberta-base" )

it=smoothgrad
for n in "${!models[@]}"; do
    echo $n ${models[n]} ${lname[n]}

    CUDA_VISIBLE_DEVICES=${gpu} nohup python interpret_and_save.py --dataset_name $dataset_name --dataset_root $dataset_root --model_name ${models[n]} --it $it --save_eval_result 1 --it_configs "{\"noise_amount\": 0.1, \"n_samples\": 50, \"max_seq_len\": 512}" >> output/$it-${models[n]}.log &
    pid[$gpu]=$!
    echo $gpu start: pid=$!
    gpu=$((gpu+1))
    if [[ $gpu -eq 8 ]]; then gpu=0; echo "8 tasks are running."; wait; fi
done



it=intgrad
for n in "${!models[@]}"; do
    echo $n ${models[n]} ${lname[n]}

    CUDA_VISIBLE_DEVICES=${gpu} nohup python interpret_and_save.py --dataset_name $dataset_name --dataset_root $dataset_root --model_name ${models[n]} --it $it --save_eval_result 1 --it_configs "{\"steps\": 50, \"max_seq_len\": 512}" >> output/$it-${models[n]}.log &
    pid[$gpu]=$!
    echo $gpu start: pid=$!
    gpu=$((gpu+1))
    if [[ $gpu -eq 8 ]]; then gpu=0; echo "8 tasks are running."; wait; fi
done


it=lime
for n in "${!models[@]}"; do
    echo $n ${models[n]} ${lname[n]}

    CUDA_VISIBLE_DEVICES=${gpu} nohup python interpret_and_save.py --dataset_name $dataset_name --dataset_root $dataset_root --model_name ${models[n]} --it $it --save_eval_result 1 --it_configs "{\"num_samples\": 1000, \"max_seq_len\": 512}" >> output/$it-${models[n]}.log &
    pid[$gpu]=$!
    echo $gpu start: pid=$!
    gpu=$((gpu+1))
    if [[ $gpu -eq 8 ]]; then gpu=0; echo "8 tasks are running."; wait; fi
done







models=( "bert-large-uncased" )  # 24 layers

for n in "${!models[@]}"; do
    echo $n ${models[n]}
    
    model=${models[n]}
    layers=( 0 4 8 12 16 20 23 )
    for m in "${!layers[@]}"; do
        sl=${layers[m]}
        # GA
        it="ga"
        CUDA_VISIBLE_DEVICES=${gpu} nohup python interpret_and_save.py --dataset_name $dataset_name --dataset_root $dataset_root --model_name $model --it $it --save_eval_result 1 --it_configs "{\"start_layer\": ${sl}, \"max_seq_len\": 300}" >> output/${model}-${it}-${sl}.log &
        pid[$gpu]=$!
        echo $gpu start: pid=$!
        gpu=$((gpu+1))
        if [[ $gpu -eq 8 ]]; then gpu=0; echo "8 tasks are running."; wait; fi

        # BT-head
        it="bt"
        CUDA_VISIBLE_DEVICES=${gpu} nohup python interpret_and_save.py --dataset_name $dataset_name --dataset_root $dataset_root --model_name $model --it $it --save_eval_result 1 --it_configs "{\"start_layer\": ${sl}, \"ap_mode\": \"head\", \"max_seq_len\": 300}" >> output/${model}-${it}-head-${sl}.log &
        pid[$gpu]=$!
        echo $gpu start: pid=$!
        gpu=$((gpu+1))
        if [[ $gpu -eq 8 ]]; then gpu=0; echo "8 tasks are running."; wait; fi

        # BT-token
        it="bt"
        CUDA_VISIBLE_DEVICES=${gpu} nohup python interpret_and_save.py --dataset_name $dataset_name --dataset_root $dataset_root --model_name $model --it $it --save_eval_result 1 --it_configs "{\"start_layer\": ${sl}, \"ap_mode\": \"token\", \"max_seq_len\": 300}" >> output/${model}-${it}-token-${sl}.log &
        pid[$gpu]=$!
        echo $gpu start: pid=$!
        gpu=$((gpu+1))
        if [[ $gpu -eq 8 ]]; then gpu=0; echo "8 tasks are running."; wait; fi
    done        
done

it=smoothgrad
for n in "${!models[@]}"; do
    echo $n ${models[n]} ${lname[n]}

    CUDA_VISIBLE_DEVICES=${gpu} nohup python interpret_and_save.py --dataset_name $dataset_name --dataset_root $dataset_root --model_name ${models[n]} --it $it --save_eval_result 1 --it_configs "{\"noise_amount\": 0.1, \"n_samples\": 50, \"max_seq_len\": 300}" >> output/$it-${models[n]}.log &
    pid[$gpu]=$!
    echo $gpu start: pid=$!
    gpu=$((gpu+1))
    if [[ $gpu -eq 8 ]]; then gpu=0; echo "8 tasks are running."; wait; fi
done


it=intgrad
for n in "${!models[@]}"; do
    echo $n ${models[n]} ${lname[n]}

    CUDA_VISIBLE_DEVICES=${gpu} nohup python interpret_and_save.py --dataset_name $dataset_name --dataset_root $dataset_root --model_name ${models[n]} --it $it --save_eval_result 1 --it_configs "{\"steps\": 50, \"max_seq_len\": 300}" >> output/$it-${models[n]}.log &
    pid[$gpu]=$!
    echo $gpu start: pid=$!
    gpu=$((gpu+1))
    if [[ $gpu -eq 8 ]]; then gpu=0; echo "8 tasks are running."; wait; fi
done


it=lime
for n in "${!models[@]}"; do
    echo $n ${models[n]} ${lname[n]}

    CUDA_VISIBLE_DEVICES=${gpu} nohup python interpret_and_save.py --dataset_name $dataset_name --dataset_root $dataset_root --model_name ${models[n]} --it $it --save_eval_result 1 --it_configs "{\"num_samples\": 1000, \"max_seq_len\": 300}" >> output/$it-${models[n]}.log &
    pid[$gpu]=$!
    echo $gpu start: pid=$!
    gpu=$((gpu+1))
    if [[ $gpu -eq 8 ]]; then gpu=0; echo "8 tasks are running."; wait; fi
done
wait


