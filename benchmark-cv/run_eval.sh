# For MoRF,ABPC

data=/root/codespace/benchmark_data/imagenet_val_5k.txt


# MAE ViT, GA
model=mae_vit
model_weights=/root/codespace/cjm/InterpretDL-master/mae_finetune.pdparams

gpu=0

rs_path=( "ga-2_mae_vit_ga_imagenet_val_5k.npz" "ga-4_mae_vit_ga_imagenet_val_5k.npz" "ga-6_mae_vit_ga_imagenet_val_5k.npz" "ga-8_mae_vit_ga_imagenet_val_5k.npz" )

for n in "${!rs_path[@]}"; do
    echo ${rs_path[n]}
    exp_path=${rs_path[n]}
    CUDA_VISIBLE_DEVICES=$((gpu)) nohup python evaluate_faithfulness.py --name "l100" --data_list $data --model $model --model_weights $model_weights --exp_path work_dirs/$exp_path --save_eval_result 1 --eval_configs "{\"limit_number_generated_samples\": 100}" >> output/eval-l100-${exp_path}.log &
    pid[$n]=$!
    echo $gpu start: pid=$!
    gpu=$((gpu+1))
    if [[ $gpu -eq 8 ]]
    then 
        gpu=0
        echo "8 tasks are running."
        wait
    fi
done

# ViT, GA

rs_path=( "ga-2_vit_ga_imagenet_val_5k.npz" "ga-4_vit_ga_imagenet_val_5k.npz" "ga-6_vit_ga_imagenet_val_5k.npz" "ga-8_vit_ga_imagenet_val_5k.npz" )

models=( "vit" "vit" "vit" "vit" )

for n in "${!rs_path[@]}"; do
    echo ${rs_path[n]}
    exp_path=${rs_path[n]}
    CUDA_VISIBLE_DEVICES=$((gpu)) nohup python evaluate_faithfulness.py --name "l100" --data_list $data --model ${models[n]} --exp_path work_dirs/$exp_path --save_eval_result 1 --eval_configs "{\"limit_number_generated_samples\": 100}" >> output/eval-l100-${exp_path}.log &
    pid[$n]=$!
    echo $gpu start: pid=$!
    gpu=$((gpu+1))
    if [[ $gpu -eq 8 ]]
    then 
        gpu=0
        echo "8 tasks are running."
        wait
    fi
done

wait





# MAE ViT
model=mae_vit
model_weights=/root/codespace/cjm/InterpretDL-master/mae_finetune.pdparams

gpu=0

rs_path=( "bt-head-4_mae_vit_bt_imagenet_val_5k.npz" "bt-head-6_mae_vit_bt_imagenet_val_5k.npz" "bt-head-8_mae_vit_bt_imagenet_val_5k.npz" "bt-head_mae_vit_bt_imagenet_val_5k.npz" "bt-token-4_mae_vit_bt_imagenet_val_5k.npz" "bt-token-6_mae_vit_bt_imagenet_val_5k.npz" "bt-token-8_mae_vit_bt_imagenet_val_5k.npz" "bt-token_mae_vit_bt_imagenet_val_5k.npz" "default_mae_vit_bt_imagenet_val_5k.npz" "default_mae_vit_ga_imagenet_val_5k.npz" "default_mae_vit_intgrad_imagenet_val_5k.npz" "default_mae_vit_smoothgrad_imagenet_val_5k.npz" )

for n in "${!rs_path[@]}"; do
    echo ${rs_path[n]}
    exp_path=${rs_path[n]}
    CUDA_VISIBLE_DEVICES=$((gpu)) nohup python evaluate_faithfulness.py --name "l100" --data_list $data --model $model --model_weights $model_weights --exp_path work_dirs/$exp_path --save_eval_result 1 --eval_configs "{\"limit_number_generated_samples\": 100}" >> output/eval-l100-${exp_path}.log &
    pid[$n]=$!
    echo $gpu start: pid=$!
    gpu=$((gpu+1))
    if [[ $gpu -eq 8 ]]
    then 
        gpu=0
        echo "8 tasks are running."
        wait
    fi
done


# other models and exps.

rs_path=( "bt-head_vit_bt_imagenet_val_5k.npz" "bt-token_vit_bt_imagenet_val_5k.npz" "default_ResNet101_gradcam_imagenet_val_5k.npz" "default_ResNet101_intgrad_imagenet_val_5k.npz" "default_ResNet101_smoothgrad_imagenet_val_5k.npz" "default_ResNet152_gradcam_imagenet_val_5k.npz" "default_ResNet152_intgrad_imagenet_val_5k.npz" "default_ResNet152_smoothgrad_imagenet_val_5k.npz" "default_ResNet50_gradcam_imagenet_val_5k.npz" "default_ResNet50_intgrad_imagenet_val_5k.npz" "default_ResNet50_smoothgrad_imagenet_val_5k.npz" "default_mobilenet_gradcam_imagenet_val_5k.npz" "default_mobilenet_intgrad_imagenet_val_5k.npz" "default_mobilenet_smoothgrad_imagenet_val_5k.npz" "default_vit_bt_imagenet_val_5k.npz" "default_vit_ga_imagenet_val_5k.npz" "default_vit_intgrad_imagenet_val_5k.npz" "default_vit_smoothgrad_imagenet_val_5k.npz" )

models=( "vit" "vit" "ResNet101" "ResNet101" "ResNet101" "ResNet152" "ResNet152" "ResNet152" "ResNet50" "ResNet50" "ResNet50" "mobilenet" "mobilenet" "mobilenet" "vit" "vit" "vit" "vit" )

for n in "${!rs_path[@]}"; do
    echo ${rs_path[n]}
    exp_path=${rs_path[n]}
    CUDA_VISIBLE_DEVICES=$((gpu)) nohup python evaluate_faithfulness.py --name "l100" --data_list $data --model ${models[n]} --exp_path work_dirs/$exp_path --save_eval_result 1 --eval_configs "{\"limit_number_generated_samples\": 100}" >> output/eval-l100-${exp_path}.log &
    pid[$n]=$!
    echo $gpu start: pid=$!
    gpu=$((gpu+1))
    if [[ $gpu -eq 8 ]]
    then 
        gpu=0
        echo "8 tasks are running."
        wait
    fi
done
wait
