cd ..    # switch to root dir

python ./train.py --dataset CUB --method jigsaw --test_n_way 5 --n_shot 5 --gpu 3
python ./test.py --dataset CUB --method jigsaw --test_n_way 5 --n_shot 1 --gpu 3
python ./test.py --dataset CUB --method jigsaw --test_n_way 5 --n_shot 5 --gpu 3

python ./train.py --dataset CUB --method imprint_jigsaw --test_n_way 5 --n_shot 5 --gpu 3
python ./test.py --dataset CUB --method imprint_jigsaw --test_n_way 5 --n_shot 1 --gpu 3
python ./test.py --dataset CUB --method imprint_jigsaw --test_n_way 5 --n_shot 5 --gpu 3