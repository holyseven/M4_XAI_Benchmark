
data=/root/codespace/benchmark-it/benchmark_data/imagenet_val_5k.txt
gpu=0

##### sg #####
models=( "resnet101" "resnet152" "resnet50" "mobilenet" "vgg16" "vit_base" "vit_large" "vit_small" )
weights=( "/root/codespace/benchmark-it/training/results/result_ResNet101_yellow/model.pd" "/root/codespace/benchmark-it/training/results/result_ResNet152_yellow/model.pd" "/root/codespace/benchmark-it/training/results/result_ResNet50_yellow/model.pd" "/root/codespace/benchmark-it/training/results/result_mobilenet_yellow/model.pd" "/root/codespace/benchmark-it/training/results/result_vgg16_yellow/model.pd" "/root/codespace/benchmark-it/training/results/result_vit_base_yellow/model.pd" "/root/codespace/benchmark-it/training/results/result_vit_large_yellow/model.pd" "/root/codespace/benchmark-it/training/results/result_vit_small_yellow/model.pd" )
it=smoothgrad
for n in "${!models[@]}"; do
    model=${models[n]}
    model_weights=${weights[n]}
    CUDA_VISIBLE_DEVICES=$((gpu)) nohup python evaluate_pgs.py --data_list $data --model ${model} --num_classes 2 --model_weights ${model_weights} --it smoothgrad --it_configs "{\"noise_amount\": 0.001, \"n_samples\": 100}" --save_eval_result 1 >> output/eval-psg-${model}-${it}.log &
    pid[$n]=$!
    echo $gpu start: pid=$!
    gpu=$((gpu+1))
    if [[ $gpu -eq 8 ]]; then gpu=0; echo "8 tasks are running."; wait; fi

done


##### ig #####
models=( "resnet101" "resnet152" "resnet50" "mobilenet" "vgg16" "vit_base" "vit_large" "vit_small" )
weights=( "/root/codespace/benchmark-it/training/results/result_ResNet101_yellow/model.pd" "/root/codespace/benchmark-it/training/results/result_ResNet152_yellow/model.pd" "/root/codespace/benchmark-it/training/results/result_ResNet50_yellow/model.pd" "/root/codespace/benchmark-it/training/results/result_mobilenet_yellow/model.pd" "/root/codespace/benchmark-it/training/results/result_vgg16_yellow/model.pd" "/root/codespace/benchmark-it/training/results/result_vit_base_yellow/model.pd" "/root/codespace/benchmark-it/training/results/result_vit_large_yellow/model.pd" "/root/codespace/benchmark-it/training/results/result_vit_small_yellow/model.pd" )
it=intgrad
for n in "${!models[@]}"; do
    model=${models[n]}
    model_weights=${weights[n]}
    CUDA_VISIBLE_DEVICES=$((gpu)) nohup python evaluate_pgs.py --data_list $data --model ${model} --num_classes 2 --model_weights ${model_weights} --it intgrad --it_configs "{\"num_random_trials\": 5, \"baselines\": \"random\", \"steps\": 20}" --save_eval_result 1 >> output/eval-psg-${model}-${it}.log &
    pid[$n]=$!
    echo $gpu start: pid=$!
    gpu=$((gpu+1))
    if [[ $gpu -eq 8 ]]; then gpu=0; echo "8 tasks are running."; wait; fi

done



##### sg-sq #####
models=( "resnet101" "resnet152" "resnet50" "mobilenet" "vgg16" "vit_base" "vit_large" "vit_small" )
weights=( "/root/codespace/benchmark-it/training/results/result_ResNet101_yellow/model.pd" "/root/codespace/benchmark-it/training/results/result_ResNet152_yellow/model.pd" "/root/codespace/benchmark-it/training/results/result_ResNet50_yellow/model.pd" "/root/codespace/benchmark-it/training/results/result_mobilenet_yellow/model.pd" "/root/codespace/benchmark-it/training/results/result_vgg16_yellow/model.pd" "/root/codespace/benchmark-it/training/results/result_vit_base_yellow/model.pd" "/root/codespace/benchmark-it/training/results/result_vit_large_yellow/model.pd" "/root/codespace/benchmark-it/training/results/result_vit_small_yellow/model.pd" )
it=smoothgrad
for n in "${!models[@]}"; do
    model=${models[n]}
    model_weights=${weights[n]}
    CUDA_VISIBLE_DEVICES=$((gpu)) nohup python evaluate_pgs.py --name 'smoothgrad-sq' --data_list $data --model ${model} --num_classes 2 --model_weights ${model_weights} --it smoothgrad --it_configs "{\"noise_amount\": 0.001, \"n_samples\": 100}" --save_eval_result 1 >> output/eval-psg-${model}-${it}-sq.log &
    pid[$n]=$!
    echo $gpu start: pid=$!
    gpu=$((gpu+1))
    if [[ $gpu -eq 8 ]]; then gpu=0; echo "8 tasks are running."; wait; fi

done


##### ig-sq #####
models=( "resnet101" "resnet152" "resnet50" "mobilenet" "vgg16" "vit_base" "vit_large" "vit_small" )
weights=( "/root/codespace/benchmark-it/training/results/result_ResNet101_yellow/model.pd" "/root/codespace/benchmark-it/training/results/result_ResNet152_yellow/model.pd" "/root/codespace/benchmark-it/training/results/result_ResNet50_yellow/model.pd" "/root/codespace/benchmark-it/training/results/result_mobilenet_yellow/model.pd" "/root/codespace/benchmark-it/training/results/result_vgg16_yellow/model.pd" "/root/codespace/benchmark-it/training/results/result_vit_base_yellow/model.pd" "/root/codespace/benchmark-it/training/results/result_vit_large_yellow/model.pd" "/root/codespace/benchmark-it/training/results/result_vit_small_yellow/model.pd" )
it=intgrad
for n in "${!models[@]}"; do
    model=${models[n]}
    model_weights=${weights[n]}
    CUDA_VISIBLE_DEVICES=$((gpu)) nohup python evaluate_pgs.py --name 'intgrad-sq' --data_list $data --model ${model} --num_classes 2 --model_weights ${model_weights} --it intgrad --it_configs "{\"num_random_trials\": 5, \"baselines\": \"random\", \"steps\": 20}" --save_eval_result 1 >> output/eval-psg-${model}-${it}-sq.log &
    pid[$n]=$!
    echo $gpu start: pid=$!
    gpu=$((gpu+1))
    if [[ $gpu -eq 8 ]]; then gpu=0; echo "8 tasks are running."; wait; fi

done



############ gradcam #########

# resnet models
models=( "ResNet101" "ResNet152" "ResNet50" )
weights=( "/root/codespace/benchmark-it/training/results/result_ResNet101_yellow/model.pd" "/root/codespace/benchmark-it/training/results/result_ResNet152_yellow/model.pd" "/root/codespace/benchmark-it/training/results/result_ResNet50_yellow/model.pd" )

lname=( "layer4.2.relu" "layer3.0" "layer2.0" "layer1.0")
it=gradcam

gpu=0
for n in "${!models[@]}"; do
    model=${models[n]}
    model_weights=${weights[n]}
    for m in "${!lname[@]}"; do
        layername=${lname[m]}
        echo $n ${model} ${layername}
        CUDA_VISIBLE_DEVICES=${gpu} nohup python evaluate_pgs.py --name ${it}-${layername} --data_list $data --model ${model} --num_classes 2 --model_weights ${model_weights} --it gradcam --save_eval_result 1 --it_configs "{\"target_layer_name\": \"${layername}\"}" >> output/eval-psg-${model}-${it}-${layername}.log &
        pid[$gpu]=$!
        echo $gpu start: pid=$!
        gpu=$((gpu+1))
        if [[ $gpu -eq 8 ]]; then gpu=0; echo "8 tasks are running."; wait; fi
    done
done

# mobilenet
# lname=( "blocks.2" "blocks.4" "blocks.6" "blocks.8" "blocks.10" "blocks.12" "blocks.14" )
lname=( "blocks.2" "blocks.8" "blocks.14" )
model='mobilenet'
model_weights="/root/codespace/benchmark-it/training/results/result_mobilenet_yellow/model.pd"
for m in "${!lname[@]}"; do
    layername=${lname[m]}
    echo $n ${model} ${layername}
    CUDA_VISIBLE_DEVICES=${gpu} nohup python evaluate_pgs.py --name ${it}-${layername} --data_list $data --model ${model} --num_classes 2 --model_weights ${model_weights} --it gradcam --save_eval_result 1 --it_configs "{\"target_layer_name\": \"${layername}\"}" >> output/eval-psg-${model}-${it}-${layername}.log &
    pid[$gpu]=$!
    echo $gpu start: pid=$!
    gpu=$((gpu+1))
    if [[ $gpu -eq 8 ]]; then gpu=0; echo "8 tasks are running."; wait; fi
done

# vgg16
lname=( "features.10" "features.20" "features.30" )
model='vgg16'
model_weights="/root/codespace/benchmark-it/training/results/result_vgg16_yellow/model.pd"
for m in "${!lname[@]}"; do
    layername=${lname[m]}
    echo $n ${model} ${layername}
    CUDA_VISIBLE_DEVICES=${gpu} nohup python evaluate_pgs.py --name ${it}-${layername} --data_list $data --model ${model} --num_classes 2 --model_weights ${model_weights} --it gradcam --save_eval_result 1 --it_configs "{\"target_layer_name\": \"${layername}\"}" >> output/eval-psg-${model}-${it}-${layername}.log &
    pid[$gpu]=$!
    echo $gpu start: pid=$!
    gpu=$((gpu+1))
    if [[ $gpu -eq 8 ]]; then gpu=0; echo "8 tasks are running."; wait; fi
done



####### ga #########

models=( "vit_base" "vit_large" "vit_small" )
weights=( "/root/codespace/benchmark-it/training/results/result_vit_base_yellow/model.pd" "/root/codespace/benchmark-it/training/results/result_vit_large_yellow/model.pd" "/root/codespace/benchmark-it/training/results/result_vit_small_yellow/model.pd" )
layers=( 4 8 2 )
it=ga
for n in "${!models[@]}"; do
    echo $n ${models[n]}
    model=${models[n]}
    model_weights=${weights[n]}
    sl=${layers[n]}
    CUDA_VISIBLE_DEVICES=${gpu} nohup python evaluate_pgs.py --name ${it}-${sl} --data_list $data --model ${model} --num_classes 2 --model_weights ${model_weights} --it ga --save_eval_result 1 --it_configs "{\"start_layer\": ${sl}}" >> output/eval-psg-${model}-${it}.log &
    pid[$gpu]=$!
    echo $gpu start: pid=$!
    gpu=$((gpu+1))
    if [[ $gpu -eq 8 ]]; then gpu=0; echo "8 tasks are running."; wait; fi
done




####### bt #########

# BT-head

models=( "vit_base" "vit_large" "vit_small" )
weights=( "/root/codespace/benchmark-it/training/results/result_vit_base_yellow/model.pd" "/root/codespace/benchmark-it/training/results/result_vit_large_yellow/model.pd" "/root/codespace/benchmark-it/training/results/result_vit_small_yellow/model.pd" )
layers=( 4 8 2 )
it=bt
for n in "${!models[@]}"; do
    echo $n ${models[n]}
    model=${models[n]}
    model_weights=${weights[n]}
    sl=${layers[n]}
    
    CUDA_VISIBLE_DEVICES=${gpu} nohup python evaluate_pgs.py --name ${it}-head --data_list $data --model ${model} --num_classes 2 --model_weights ${model_weights} --it bt --save_eval_result 1 --it_configs "{\"start_layer\": ${sl}, \"ap_mode\": \"head\"}" >> output/eval-psg-${model}-${it}-head.log &    

    pid[$gpu]=$!
    echo $gpu start: pid=$!
    gpu=$((gpu+1))
    if [[ $gpu -eq 8 ]]; then gpu=0; echo "8 tasks are running."; wait; fi
done


models=( "vit_base" "vit_large" "vit_small" )
weights=( "/root/codespace/benchmark-it/training/results/result_vit_base_yellow/model.pd" "/root/codespace/benchmark-it/training/results/result_vit_large_yellow/model.pd" "/root/codespace/benchmark-it/training/results/result_vit_small_yellow/model.pd" )
layers=( 4 8 2 )
it=bt
for n in "${!models[@]}"; do
    echo $n ${models[n]}
    model=${models[n]}
    model_weights=${weights[n]}
    sl=${layers[n]}
    
    CUDA_VISIBLE_DEVICES=${gpu} nohup python evaluate_pgs.py --name ${it}-token --data_list $data --model ${model} --num_classes 2 --model_weights ${model_weights} --it bt --save_eval_result 1 --it_configs "{\"start_layer\": ${sl}, \"ap_mode\": \"token\"}" >> output/eval-psg-${model}-${it}-token.log &    

    pid[$gpu]=$!
    echo $gpu start: pid=$!
    gpu=$((gpu+1))
    if [[ $gpu -eq 8 ]]; then gpu=0; echo "8 tasks are running."; wait; fi
done



###### mae #######
model="mae_vit"
model_weights="/root/codespace/benchmark-it/training/results/result_mae_vit_yellow/model.pd"

# bt
it=bt
sl=4

CUDA_VISIBLE_DEVICES=${gpu} nohup python evaluate_pgs.py --name ${it}-token --data_list $data --model ${model} --num_classes 2 --model_weights ${model_weights} --it bt --save_eval_result 1 --it_configs "{\"start_layer\": ${sl}, \"ap_mode\": \"token\"}" >> output/eval-psg-${model}-${it}-token.log &    
pid[$gpu]=$!
echo $gpu start: pid=$!
gpu=$((gpu+1))
if [[ $gpu -eq 8 ]]; then gpu=0; echo "8 tasks are running."; wait; fi

CUDA_VISIBLE_DEVICES=${gpu} nohup python evaluate_pgs.py --name ${it}-head --data_list $data --model ${model} --num_classes 2 --model_weights ${model_weights} --it bt --save_eval_result 1 --it_configs "{\"start_layer\": ${sl}, \"ap_mode\": \"head\"}" >> output/eval-psg-${model}-${it}-head.log &    
pid[$gpu]=$!
echo $gpu start: pid=$!
gpu=$((gpu+1))
if [[ $gpu -eq 8 ]]; then gpu=0; echo "8 tasks are running."; wait; fi


# ga
it=ga
sl=4

CUDA_VISIBLE_DEVICES=${gpu} nohup python evaluate_pgs.py --name ${it}-${sl} --data_list $data --model ${model} --num_classes 2 --model_weights ${model_weights} --it ga --save_eval_result 1 --it_configs "{\"start_layer\": ${sl}}" >> output/eval-psg-${model}-${it}.log &
pid[$gpu]=$!
echo $gpu start: pid=$!
gpu=$((gpu+1))
if [[ $gpu -eq 8 ]]; then gpu=0; echo "8 tasks are running."; wait; fi

##### sg #####
it=smoothgrad

CUDA_VISIBLE_DEVICES=$((gpu)) nohup python evaluate_pgs.py --data_list $data --model ${model} --num_classes 2 --model_weights ${model_weights} --it smoothgrad --it_configs "{\"noise_amount\": 0.001, \"n_samples\": 100}" --save_eval_result 1 >> output/eval-psg-${model}-${it}.log &
pid[$n]=$!
echo $gpu start: pid=$!
gpu=$((gpu+1))
if [[ $gpu -eq 8 ]]; then gpu=0; echo "8 tasks are running."; wait; fi

CUDA_VISIBLE_DEVICES=$((gpu)) nohup python evaluate_pgs.py --name 'smoothgrad-sq' --data_list $data --model ${model} --num_classes 2 --model_weights ${model_weights} --it smoothgrad --it_configs "{\"noise_amount\": 0.001, \"n_samples\": 100}" --save_eval_result 1 >> output/eval-psg-${model}-${it}-sq.log &
pid[$n]=$!
echo $gpu start: pid=$!
gpu=$((gpu+1))
if [[ $gpu -eq 8 ]]; then gpu=0; echo "8 tasks are running."; wait; fi


##### ig #####
it=intgrad

CUDA_VISIBLE_DEVICES=$((gpu)) nohup python evaluate_pgs.py --data_list $data --model ${model} --num_classes 2 --model_weights ${model_weights} --it intgrad --it_configs "{\"num_random_trials\": 5, \"baselines\": \"random\", \"steps\": 20}" --save_eval_result 1 >> output/eval-psg-${model}-${it}.log &
pid[$n]=$!
echo $gpu start: pid=$!
gpu=$((gpu+1))
if [[ $gpu -eq 8 ]]; then gpu=0; echo "8 tasks are running."; wait; fi

CUDA_VISIBLE_DEVICES=$((gpu)) nohup python evaluate_pgs.py --name 'intgrad-sq' --data_list $data --model ${model} --num_classes 2 --model_weights ${model_weights} --it intgrad --it_configs "{\"num_random_trials\": 5, \"baselines\": \"random\", \"steps\": 20}" --save_eval_result 1 >> output/eval-psg-${model}-${it}-sq.log &
pid[$n]=$!
echo $gpu start: pid=$!
gpu=$((gpu+1))
if [[ $gpu -eq 8 ]]; then gpu=0; echo "8 tasks are running."; wait; fi
