# ContinualLocalReplacement
Continual local replacement is a simple yet effective method for few-shot image recognition.
This method is able to introduce more semantic information and significantly enlarge the supervised signals in embedded space for decision boundary learning. See T-SNE visualization and saliency maps below.
In T-SNE visualization, the labeled images (i.e. support set) and synthesized images are marked by stars and triangles, respectively.
During fine tuning, the embedded features change and can cover a larger area within each embedded cluster for learning.


This algorithm is based on standard transfer learning and thus it has a good scalability.
For example, it can be easily combined with the weight imprinting technique.

This repository includes the source codes of ContinualLocalReplacement and its weight imprinting variant.
Please check our published paper for more algorithm details (https://arxiv.org/abs/2001.08366).

<div align="center">
<img src="https://raw.githubusercontent.com/Lecanyu/ContinualLocalReplacement/master/images/tsne_visualization2.gif" height="300px" alt="demo2" >
<img src="https://raw.githubusercontent.com/Lecanyu/ContinualLocalReplacement/master/images/tsne_visualization1.gif" height="300px" alt="demo1" >
</div>

![demo3](https://raw.githubusercontent.com/Lecanyu/ContinualLocalReplacement/master/images/saliency_map.png)


# 1. Prerequisites
This code is implemented on Pytorch. 
So you may need to install
* Python 3.x
* Pytorch 1.0.0 or above

We have tested this code on Ubuntu and Windows10 x64 operation system.


# 2. Run 
Run codes using below commands. 
We conduct all experiments on default hyper-parameters. But you may try other hyper-parameters.
For example, you can actually achieve better performance than the reported results when you test imprint_jigsaw method with jig_replace_max_num=9 on CUB dataset.

We also provide the driver scripts. Please check the run_scripts folder.

Organize Dataset
------------
Before runing, please check the code to make sure all pathes set correctly.
```
python ./filelists/miniImagenet/write_data_json.py
python ./filelists/tieredImagenet/write_data_json.py
python ./filelists/CUB/write_data_json.py
python ./filelists/caltech256/write_data_json.py
```


Training
------------
```
python ./train.py --dataset miniImagenet/tieredImagenet/CUB/caltech256 --method jigsaw --test_n_way 5 --n_shot 5 --gpu 0
python ./train.py --dataset miniImagenet/tieredImagenet/CUB/caltech256 --method imprint_jigsaw --test_n_way 5 --n_shot 5 --gpu 0
```

Testing
------------
```
python ./test.py --dataset miniImagenet/tieredImagenet/CUB/caltech256 --method jigsaw --test_n_way 5 --n_shot 1 --gpu 1
python ./test.py --dataset miniImagenet/tieredImagenet/CUB/caltech256 --method jigsaw --test_n_way 5 --n_shot 5 --gpu 1
python ./test.py --dataset miniImagenet/tieredImagenet/CUB/caltech256 --method imprint_jigsaw --test_n_way 5 --n_shot 1 --gpu 1
python ./test.py --dataset miniImagenet/tieredImagenet/CUB/caltech256 --method imprint_jigsaw --test_n_way 5 --n_shot 5 --gpu 1
```

# 3. Datasets and Pre-trained Parameters
We provide our pre-trained parameters for reproducing the results in the paper.
You can download them from [HERE](http://graphics.xmu.edu.cn/students/canyu/checkpoints_for_paper_results.zip).
 
To make it convenient, we also provide links for downloading the experiment datasets ([MiniImagenet](http://graphics.xmu.edu.cn/students/canyu/miniImagenet.zip), [TieredImagenet](http://graphics.xmu.edu.cn/students/canyu/tiered_imagenet.tar), [CUB-200](http://graphics.xmu.edu.cn/students/canyu/CUB200.tgz), [Caltech-256](http://graphics.xmu.edu.cn/students/canyu/caltech256.tar)) even though you can download them from original sources or official websites.

# 4. Results
|          (%)           | Mini 1-shot  | Mini 5-shot  | Tiered 1-shot | Tiered 5-shot | 
| ---------------------- | ------------ | ------------ | ------------ | ------------ |
| Vanilla           | `56.44 ± 0.81` | `78.18 ± 0.60` | `63.93 ± 0.89` | `84.66 ± 0.62` |
| CLR           | __`66.33 ± 0.93`__ | __`81.12 ± 0.60`__ | __`74.76 ± 0.97`__ | __`86.78 ± 0.66`__ | 
| CLR + Imprinting           | `63.54 ± 0.85` | `78.26 ± 0.61` | `73.29 ± 0.97` | `85.04 ± 0.64` | 

|          (%)           | CUB-200 1-shot | CUB-200 5-shot  | Caltech256 1-shot | Caltech256 5-shot | 
| ---------------------- | ------------ | ------------ | ------------ | ------------ |
| Vanilla                | `65.91 ± 0.88` | `83.45 ± 0.51` | `59.32 ± 0.88` | `79.95 ± 0.67` |
| CLR                    | `68.00 ± 0.92` | `84.26 ± 0.53` | __`65.74 ± 0.92`__ | __`83.05 ± 0.60`__ | 
| CLR + Imprinting       | __`73.80 ± 0.92`__ | __`87.46 ± 0.48`__ | __`65.88 ± 0.97`__ | `81.16 ± 0.64` | 


# 5. Citation
If our work is useful in your research, please cite 

```
@article{le2020local,
  title={Continual Local Replacement for Few-shot Image Recognition},
  author={Canyu Le, Zhonggui Chen, Xihan Wei, Biao Wang, Lei Zhang},
  journal={arXiv preprint arXiv:2001.08366},
  year={2020}
}
```

# 5. References
This implementation builds upon several open-souce codes.
Specifically, we have modified and integrated the following codes into this repository:

* CloserLookFewShot https://github.com/wyharveychen/CloserLookFewShot
* Weight Imprinting https://github.com/YU1ut/imprinted-weights



