# ContinualLocalReplacement
Continual local replacement is a simple yet effective method for few-shot image recognition.
This method is able to introduce more semantic information and significantly enlarge the supervised signals in embedded space for decision boundary learning. See T-SNE visualization and saliency maps below.

This algorithm is based on standard transfer learning and thus it has a good scalability.
For example, it can be easily combined with the weight imprinting technique.

This repository includes the source codes of ContinualLocalReplacement and its weight imprinting variant.
Please check our published paper for more algorithm details (XXX).

<div align="center">
<img src="https://github.com/Lecanyu/ContinualLocalReplacement/raw/master/images/tsne_visualization2.gif" height="300px" alt="demo2" >
<img src="https://github.com/Lecanyu/ContinualLocalReplacement/raw/master/images/tsne_visualization1.gif" height="300px" alt="demo1" >
</div>

![demo3](https://github.com/Lecanyu/ContinualLocalReplacement/raw/master/images/saliency_map.png)


# 1. Prerequisites
This code is implemented on Pytorch. 
So you may need to install
* Python 3.x
* Pytorch 1.0.0 or above

We have tested this code on Ubuntu and Windows10 x64 operation system.


# 2. Run 

Training
------------

Testing
------------


# 3. Datasets and pre-trained net parameters
We provide our pre-trained parameters for reproducing the results in the paper.
You can download them from [here](xxx).

To make it convenient, we also provide links for downloading the experiment datasets ([MiniImagenet](xxx), [TieredImagenet](xxx), [CUB-200](xxx), [Caltech-256](xxx)) even though people can download them from original sources or official websites.


# 4. Citation
If our work is useful in your research, please cite 

```
Will do
```

# 5. References
This implementation builds upon several open-souce codes.
Specifically, we have modified and integrated the following codes into this repository:

* CloserLookFewShot https://github.com/wyharveychen/CloserLookFewShot
* Weight Imprinting https://github.com/YU1ut/imprinted-weights



