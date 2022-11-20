# Turn off tokenizer parallelism
export TOKENIZERS_PARALLELISM=false

python train.py\
    --dataset_name="imdb"\
    --batch_sz=5\
    --n_epochs=10\
    --n_iters_per_train_epoch=10000\
    --run_test=True

python train.py\
    --dataset_name="sst5"\
    --batch_sz=50\
    --n_epochs=10\
    --n_iters_per_train_epoch=2000\
    --run_test=True

python train.py\
    --pretrained_path="checkpoints/sst5_head_for_roberta-large_epoch9.checkpoint"\
    --dataset_name="sst2"\
    --batch_sz=50\
    --n_epochs=3\
    --n_iters_per_train_epoch=2000\
    --run_test=True

python train.py\
    --dataset_name="imagenet-1k"\
    --batch_sz=50\
    --n_epochs=10\
    --run_test=True

python train.py\
    --dataset_name="cifar10"\
    --batch_sz=50\
    --n_epochs=10\
    --n_iters_per_train_epoch=2000\
    --run_test=True

python train.py\
    --dataset_name="cifar100"\
    --batch_sz=50\
    --n_epochs=10\
    --n_iters_per_train_epoch=2000\
    --run_test=True