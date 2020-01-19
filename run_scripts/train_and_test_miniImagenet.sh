cd ..    # switch to root dir

python ./train.py --dataset miniImagenet --method jigsaw --test_n_way 5 --n_shot 5 --gpu 1
python ./test.py --dataset miniImagenet --method jigsaw --test_n_way 5 --n_shot 1 --gpu 1
python ./test.py --dataset miniImagenet --method jigsaw --test_n_way 5 --n_shot 5 --gpu 1

python ./train.py --dataset miniImagenet --method imprint_jigsaw --test_n_way 5 --n_shot 5 --gpu 1
python ./test.py --dataset miniImagenet --method imprint_jigsaw --test_n_way 5 --n_shot 1 --gpu 1
python ./test.py --dataset miniImagenet --method imprint_jigsaw --test_n_way 5 --n_shot 5 --gpu 1




