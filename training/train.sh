export LD_LIBRARY_PATH=/opt/conda/envs/pp241/lib:$LD_LIBRARY_PATH 

# should be changed.
dataset_root=/root/codespace/xai_benchmark/benchmark_data/imagenet_train_10k
data=/root/codespace/xai_benchmark/benchmark_data/imagenet_train_10k.txt
val_dataset_root=/root/codespace/xai_benchmark/benchmark_data/imagenet_val_5k
val_data=/root/codespace/xai_benchmark/benchmark_data/imagenet_val_5k.txt

gpu=6

models=( "ResNet50" "ResNet101" "ResNet152" )

for n in "${!models[@]}"; do
    model=${models[n]}
    name=${model}_yellow
    CUDA_VISIBLE_DEVICES=${gpu} python train_model.py --name $name --model_name $model --data_list $data --dataset_root $dataset_root --val_dataset_root $val_dataset_root --val_data_list $val_data --lr 0.01 --num_classes 2 --epochs 3
done

models=( "vgg16" )

for n in "${!models[@]}"; do
    model=${models[n]}
    name=${model}_yellow
    CUDA_VISIBLE_DEVICES=${gpu} python train_model.py --name $name --model_name $model --data_list $data --dataset_root $dataset_root --val_dataset_root $val_dataset_root --val_data_list $val_data --lr 0.001 --num_classes 2 --epochs 3
done

models=( "mobilenet" )

for n in "${!models[@]}"; do
    model=${models[n]}
    name=${model}_yellow
    CUDA_VISIBLE_DEVICES=${gpu} python train_model.py --name $name --model_name $model --data_list $data --dataset_root $dataset_root --val_dataset_root $val_dataset_root --val_data_list $val_data --lr 0.1 --num_classes 2 --epochs 3
done


models=( "vit_base" "vit_small" "vit_large" )

for n in "${!models[@]}"; do
    model=${models[n]}
    name=${model}_yellow
    CUDA_VISIBLE_DEVICES=${gpu} python train_model.py --name $name --model_name $model --data_list $data --dataset_root $dataset_root --val_dataset_root $val_dataset_root --val_data_list $val_data --optimizer AdamW --lr 0.001 --num_classes 2 --epochs 3
done


models=( "mae_vit" )

for n in "${!models[@]}"; do
    model=${models[n]}
    name=${model}_yellow
    CUDA_VISIBLE_DEVICES=${gpu} python train_model.py --name $name --model_name $model --data_list $data --dataset_root $dataset_root --val_dataset_root $val_dataset_root --val_data_list $val_data --optimizer AdamW --lr 0.001 --num_classes 2 --epochs 3
done

