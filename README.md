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
Run s3dis_active.py for training, and run s3dis_test.py for test.

## Citation
````
@article{yao2025active,
  title={Active Learning with Joint Probabilistic Modeling for Point Cloud Semantic Segmentation},
  author={Yao, Baochen and Zhang, Dongjie and Zhao, Jie and Zheng, Ye and Peng, Chengbin},
  journal={Knowledge-Based Systems},
  pages={114171},
  year={2025},
  publisher={Elsevier}
}
````


## Acknowledgement
The code is built on PointNeXt. We thank the authors for sharing the code.
