
############ smoothgrad #########
data=/root/codespace/benchmark_data/imagenet_val_5k.txt
models=( "ResNet50" "ResNet101" "ResNet152" "mobilenet" "vgg16" "vit_base" "vit_small" "vit_large" )
it=smoothgrad

gpu=0
for n in "${!models[@]}"; do
    echo $n ${models[n]} ${lname[n]}
    CUDA_VISIBLE_DEVICES=${gpu} nohup python interpret_and_save.py --data_list $data --model ${models[n]} --it $it --save_eval_result 1 --it_configs "{\"noise_amount\": 0.1, \"n_samples\": 100}" >> output/$it.log &
    pid[$gpu]=$!
    # manually and naively do the job scheduling for local GPUs.
    echo $gpu start: pid=$!
    gpu=$((gpu+1))
    if [[ $gpu -eq 8 ]]; then gpu=0; echo "8 tasks are running."; wait; fi
done
wait

############ intgrad #########
data=/root/codespace/benchmark_data/imagenet_val_5k.txt
models=( "ResNet50" "ResNet101" "ResNet152" "mobilenet" "vgg16" "vit_base" "vit_small" "vit_large" )
it=intgrad

gpu=0
for n in "${!models[@]}"; do
    echo $n ${models[n]} ${lname[n]}
    CUDA_VISIBLE_DEVICES=${gpu} nohup python interpret_and_save.py --data_list $data --model ${models[n]} --it $it --save_eval_result 1 --it_configs "{\"num_random_trials\": 10, \"baselines\": \"random\", \"steps\": 50}" >> output/$it.log &
    pid[$gpu]=$!
    # manually and naively do the job scheduling for local GPUs.
    echo $gpu start: pid=$!
    gpu=$((gpu+1))
    if [[ $gpu -eq 8 ]]; then gpu=0; echo "8 tasks are running."; wait; fi    
done
wait


############ gradcam #########
data=/root/codespace/benchmark_data/imagenet_val_5k.txt
models=( "ResNet50" "ResNet101" "ResNet152" "mobilenet" "vgg16" )
lname=( "layer4.2.relu" "layer4.2.relu" "layer4.2.relu" "lastconv.0" "features")
it=gradcam

gpu=0
for n in "${!models[@]}"; do
    echo $n ${models[n]} ${lname[n]}
    CUDA_VISIBLE_DEVICES=${gpu} nohup python interpret_and_save.py --data_list $data --model ${models[n]} --it $it --save_eval_result 1 --it_configs "{\"target_layer_name\": \"${lname[n]}\"}" >> output/$it.log &
    pid[$gpu]=$!
    # manually and naively do the job scheduling for local GPUs.
    echo $gpu start: pid=$!
    gpu=$((gpu+1))
    if [[ $gpu -eq 8 ]]; then gpu=0; echo "8 tasks are running."; wait; fi
done
wait


############ vit and ga,bt ###########
data=/root/codespace/benchmark_data/imagenet_val_5k.txt

models=( "vit_base" "vit_small" "vit_large" )
layers=( 4 2 8 )

gpu=0
for n in "${!models[@]}"; do
    echo $n ${models[n]} ${layers[n]}
    model=${models[n]}
    sl=${layers[n]}

    # GA
    it="ga"
    name=${it}-${sl}
    CUDA_VISIBLE_DEVICES=${gpu} nohup python interpret_and_save.py --name ${name} --data_list $data --model $model --it $it --save_eval_result 1 --it_configs "{\"start_layer\": ${sl}}" >> output/${model}-${name}.log &
    pid[$gpu]=$!
    # manually and naively do the job scheduling for local GPUs.
    echo $gpu start: pid=$!
    gpu=$((gpu+1))
    if [[ $gpu -eq 8 ]]; then gpu=0; echo "8 tasks are running."; wait; fi

    # BT-head
    it="bt"
    name=${it}-head-${sl}
    CUDA_VISIBLE_DEVICES=${gpu} nohup python interpret_and_save.py --name $name --data_list $data --model $model --it $it --save_eval_result 1 --it_configs "{\"start_layer\": ${sl}, \"ap_mode\": \"head\"}" >> output/${model}-${name}.log &
    pid[$gpu]=$!
    # manually and naively do the job scheduling for local GPUs.
    echo $gpu start: pid=$!
    gpu=$((gpu+1))
    if [[ $gpu -eq 8 ]]; then gpu=0; echo "8 tasks are running."; wait; fi

    # BT-token
    it="bt"
    name=${it}-token-${sl}
    CUDA_VISIBLE_DEVICES=${gpu} nohup python interpret_and_save.py --name $name --data_list $data --model $model --it $it --save_eval_result 1 --it_configs "{\"start_layer\": ${sl}, \"ap_mode\": \"token\"}" >> output/${model}-${name}.log &
    pid[$gpu]=$!
    # manually and naively do the job scheduling for local GPUs.
    echo $gpu start: pid=$!
    gpu=$((gpu+1))
    if [[ $gpu -eq 8 ]]; then gpu=0; echo "8 tasks are running."; wait; fi 
done


############ mae and all expl #########

data=/root/codespace/benchmark_data/imagenet_val_5k.txt
model=mae_vit
model_weights=/root//codespace/cjm/InterpretDL-master/mae_finetune.pdparams

# GA
gpu=0
it="ga"
CUDA_VISIBLE_DEVICES=${gpu} nohup python interpret_and_save.py --name 'ga-2' --data_list $data --model $model --it $it --save_eval_result 1 --it_configs "{\"start_layer\": 2}" --model_weights $model_weights >> output/mae-ga-2.log &
pid[$gpu]=$!
echo $gpu start: pid=$!

gpu=1
it="ga"
CUDA_VISIBLE_DEVICES=${gpu} nohup python interpret_and_save.py --name 'ga-4' --data_list $data --model $model --it $it --save_eval_result 1 --it_configs "{\"start_layer\": 4}" --model_weights $model_weights >> output/mae-ga-4.log &
pid[$gpu]=$!
echo $gpu start: pid=$!

gpu=2
it="ga"
CUDA_VISIBLE_DEVICES=${gpu} nohup python interpret_and_save.py --name 'ga-6' --data_list $data --model $model --it $it --save_eval_result 1 --it_configs "{\"start_layer\": 6}" --model_weights $model_weights >> output/mae-ga-6.log &
pid[$gpu]=$!
echo $gpu start: pid=$!

gpu=3
it="ga"
CUDA_VISIBLE_DEVICES=${gpu} nohup python interpret_and_save.py --name 'ga-8' --data_list $data --model $model --it $it --save_eval_result 1 --it_configs "{\"start_layer\": 8}" --model_weights $model_weights >> output/mae-ga-8.log &
pid[$gpu]=$!
echo $gpu start: pid=$!


# BT
gpu=4
it="bt"
CUDA_VISIBLE_DEVICES=${gpu} nohup python interpret_and_save.py --name 'bt-head-4' --data_list $data --model $model --it $it --save_eval_result 1 --it_configs "{\"start_layer\": 4, \"ap_mode\": \"head\"}" --model_weights $model_weights >> output/mae-bt-head-4.log &
pid[$gpu]=$!
echo $gpu start: pid=$!

gpu=5
it="bt"
CUDA_VISIBLE_DEVICES=${gpu} nohup python interpret_and_save.py --name 'bt-token-4' --data_list $data --model $model --it $it --save_eval_result 1 --it_configs "{\"start_layer\": 4, \"ap_mode\": \"token\"}" --model_weights $model_weights >> output/mae-bt-token-4.log &
pid[$gpu]=$!
echo $gpu start: pid=$!

gpu=6
it="bt"
CUDA_VISIBLE_DEVICES=${gpu} nohup python interpret_and_save.py --name 'bt-head-6' --data_list $data --model $model --it $it --save_eval_result 1 --it_configs "{\"start_layer\": 6, \"ap_mode\": \"head\"}" --model_weights $model_weights >> output/mae-bt-head-6.log &
pid[$gpu]=$!
echo $gpu start: pid=$!

gpu=7
it="bt"
CUDA_VISIBLE_DEVICES=${gpu} nohup python interpret_and_save.py --name 'bt-token-6' --data_list $data --model $model --it $it --save_eval_result 1 --it_configs "{\"start_layer\": 6, \"ap_mode\": \"token\"}" --model_weights $model_weights >> output/mae-bt-token-6.log &
pid[$gpu]=$!
echo $gpu start: pid=$!

gpu=0
it="bt"
CUDA_VISIBLE_DEVICES=${gpu} nohup python interpret_and_save.py --name 'bt-head-8' --data_list $data --model $model --it $it --save_eval_result 1 --it_configs "{\"start_layer\": 8, \"ap_mode\": \"head\"}" --model_weights $model_weights >> output/mae-bt-head-8.log &
pid[$gpu]=$!
echo $gpu start: pid=$!

gpu=1
it="bt"
CUDA_VISIBLE_DEVICES=${gpu} nohup python interpret_and_save.py --name 'bt-token-8' --data_list $data --model $model --it $it --save_eval_result 1 --it_configs "{\"start_layer\": 8, \"ap_mode\": \"token\"}" --model_weights $model_weights >> output/mae-bt-token-8.log &
pid[$gpu]=$!
echo $gpu start: pid=$!

# SG
gpu=2
it="smoothgrad"
CUDA_VISIBLE_DEVICES=${gpu} nohup python interpret_and_save.py --data_list $data --model $model --it $it --save_eval_result 1 --it_configs "{\"noise_amount\": 0.1, \"n_samples\": 100}" --model_weights $model_weights >> output/mae-$it.log &
pid[$gpu]=$!
echo $gpu start: pid=$!

# IG
gpu=3
it="intgrad"
CUDA_VISIBLE_DEVICES=${gpu} nohup python interpret_and_save.py --data_list $data --model $model --it $it --save_eval_result 1 --it_configs "{\"num_random_trials\": 10, \"baselines\": \"random\", \"steps\": 50}" --model_weights $model_weights >> output/mae-$it.log &
pid[$gpu]=$!
echo $gpu start: pid=$!
