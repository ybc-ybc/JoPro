# JoPro
Active Learning with Joint Probabilistic Modeling for Point Cloud Semantic Segmentation

> With advancements in sensing technologies, the demand for point cloud semantic segmentation has grown signiﬁcantly across various applications, while current deep learning-based methods rely heavily on costly, well-annotated datasets. Recently, label-eﬃcient learning strategies have been explored to reduce annotation demands, with active learning emerging as a preferred approach by selectively annotating only the most informative samples. However, existing point cloud active learning methods often depend solely on neural network softmax scores for sample selection, which can introduce bias and be aﬀected by overconﬁdence in network predictions. To overcome this limitation, we propose an active learning framework with Joint Probabilistic mod-
eling (JoPro), aiming to select unlabeled points that can provide more post-annotation information. At the core of JoPro is a novel probabilistic model that eﬃciently captures the distribution of embedded features to generate richer probabilistic representations for unlabeled data. Utilizing this probabilistic modeling, we propose a feature mixing stability metric to identify uncertain points near decision boundaries, ensuring more informative sample selection. Furthermore, a cluster-aware hybrid contrastive regularization method is incorporated to maximize the utilization of unlabeled data to enhance training of the segmentation model. Our proposed active learning framework achieves competitive results on popular benchmarks, delivering near fully supervised performance with only 1% of the annotation budget.

This code and framework are  implemented on [PointNeXt](https://github.com/guochengqian/PointNeXt).

## Environment and Datasets
This codebase was tested with the following environment configurations.

* Ubuntu 22.04
* Python 3.7
* CUDA 11.3
* Pytorch 1.10.1

Please refer to PointNeXt to install other required packages and download datasets.

## Usage
To stabilize the training process, the first step is to train using only labeled data. Then, set this pre-trained model path in cfg_s3dis.yaml file to conduct weakly supervised training.
````
run ./UCL/main.py
````

## Citation
````
@article{yao2024uncertainty,
  title={Uncertainty-guided Contrastive Learning for Weakly Supervised Point Cloud Segmentation},
  author={Yao, Baochen and Dong, Li and Qiu, Xiaojie and Song, Kangkang and Yan, Diqun and Peng, Chengbin},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2024},
  publisher={IEEE}
}
````


## Acknowledgement
The code is built on PointNeXt. We thank the authors for sharing the code.
