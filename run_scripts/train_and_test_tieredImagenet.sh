cd ..    # switch to root dir

python ./train.py --dataset tieredImagenet --method jigsaw --test_n_way 5 --n_shot 5 --gpu 0
python ./test.py --dataset tieredImagenet --method jigsaw --test_n_way 5 --n_shot 1 --gpu 0
python ./test.py --dataset tieredImagenet --method jigsaw --test_n_way 5 --n_shot 5 --gpu 0

python ./train.py --dataset tieredImagenet --method imprint_jigsaw --test_n_way 5 --n_shot 5 --gpu 0
python ./test.py --dataset tieredImagenet --method imprint_jigsaw --test_n_way 5 --n_shot 1 --gpu 0
python ./test.py --dataset tieredImagenet --method imprint_jigsaw --test_n_way 5 --n_shot 5 --gpu 0

