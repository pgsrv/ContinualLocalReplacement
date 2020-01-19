cd ..    # switch to root dir

python ./train.py --dataset caltech256 --method jigsaw --test_n_way 5 --n_shot 5 --gpu 2
python ./test.py --dataset caltech256 --method jigsaw --test_n_way 5 --n_shot 1 --gpu 2
python ./test.py --dataset caltech256 --method jigsaw --test_n_way 5 --n_shot 5 --gpu 2

python ./train.py --dataset caltech256 --method imprint_jigsaw --test_n_way 5 --n_shot 5 --gpu 2
python ./test.py --dataset caltech256 --method imprint_jigsaw --test_n_way 5 --n_shot 1 --gpu 2
python ./test.py --dataset caltech256 --method imprint_jigsaw --test_n_way 5 --n_shot 5 --gpu 2

