
# dataset_root=/root/.paddlenlp/datasets/Glue/SST-2

# gpu=1

# # "mobilenet"

# # models=( "ResNet50" "ResNet101" "ResNet152" "mobilenet" "vgg16" )
# # models=( "ResNet50" "ResNet101" "ResNet152" )
# models=( "ernie-2.0-base-en" "bert-base-uncased" "bert-large-uncased" "distilbert-base-uncased" "roberta-base" )

# for n in "${!models[@]}"; do
#     model=${models[n]}
#     name="sst-2"
#     CUDA_VISIBLE_DEVICES=${gpu} nohup python train_model.py --name $name --model_name $model --dataset_root $dataset_root --lr 5e-6 --epochs 5 >> output/training-${name}-$model.log &
#     pid[$n]=$!
#     echo $gpu start: pid=$!
#     gpu=$((gpu+1))
#     if [[ $gpu -eq 8 ]]; then gpu=0; echo "8 tasks are running."; wait; fi
# done



dataset_root=/root/codespace/xai_benchmark/benchmark_data/movies

gpu=1

models=( "ernie-2.0-base-en" "bert-base-uncased" "distilbert-base-uncased" "roberta-base" )
for n in "${!models[@]}"; do
    model=${models[n]}
    name="movies"
    CUDA_VISIBLE_DEVICES=${gpu} nohup python train_model.py --name $name --model_name $model --dataset_root $dataset_root --batch_size 32 --lr 3e-5 --epochs 10 --max_seq_length 512 >> output/training-${name}-$model.log &
    pid[$n]=$!

    # manually and naively do the job scheduling for local GPUs.
    echo $gpu start: pid=$!
    gpu=$((gpu+1))
    if [[ $gpu -eq 8 ]]; then gpu=0; echo "8 tasks are running."; wait; fi
done

# a larger one, with a different max_seq_length.
models=( "bert-large-uncased" )

for n in "${!models[@]}"; do
    model=${models[n]}
    name="movies"
    CUDA_VISIBLE_DEVICES=${gpu} nohup python train_model.py --name $name --model_name $model --dataset_root $dataset_root --batch_size 32 --lr 3e-5 --epochs 10 --max_seq_length 300 >> output/training-${name}-$model.log &
    pid[$n]=$!

    # manually and naively do the job scheduling for local GPUs.
    echo $gpu start: pid=$!
    gpu=$((gpu+1))
    if [[ $gpu -eq 8 ]]; then gpu=0; echo "8 tasks are running."; wait; fi
done

wait
