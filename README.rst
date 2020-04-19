TResNet: High Performance GPU-Dedicated Architecture
====================================================

Packaged TResNet based on Official PyTorch Implementation
[`paper <https://arxiv.org/pdf/2003.13630.pdf>`__\ ]
[`github <https://github.com/mrT23/TResNet>`__\ ]

Installation
------------

Install with pip:

::

    pip install torch_tresnet

or directly:

::

    pip install git+https://github.com/tczhangzhi/torch-tresnet

Use
---

Follow the grammatical conventions of torchvision

::

    from torch_tresnet import tresnet_m, tresnet_l, tresnet_xl, tresnet_m_448, tresnet_l_448, tresnet_xl_448

    # pretrianed on 224*224
    model = tresnet_m(pretrain=True)
    model = tresnet_m(pretrain=True, num_classes=10)
    model = tresnet_m(pretrain=True, num_classes=10, in_chans=3)

    # pretrianed on 448*448
    model = tresnet_m_448(pretrain=True)
    model = tresnet_m_448(pretrain=True, num_classes=10)
    model = tresnet_m_448(pretrain=True, num_classes=10, in_chans=3)

Main Results
------------

TResNet Models
^^^^^^^^^^^^^^

TResNet models accuracy and GPU throughput on ImageNet, compared to
ResNet50. All measurements were done on Nvidia V100 GPU, with mixed
precision. All models are trained on input resolution of 224.

+------------------+--------------------------------+---------------------------------+------------------------+--------------+
| Models           | Top Training Speed (img/sec)   | Top Inference Speed (img/sec)   | Max Train Batch Size   | Top-1 Acc.   |
+==================+================================+=================================+========================+==============+
| ResNet50         | **805**                        | 2830                            | 288                    | 79.0         |
+------------------+--------------------------------+---------------------------------+------------------------+--------------+
| EfficientNetB1   | 440                            | 2740                            | 196                    | 79.2         |
+------------------+--------------------------------+---------------------------------+------------------------+--------------+
| TResNet-M        | 730                            | **2930**                        | **512**                | 80.7         |
+------------------+--------------------------------+---------------------------------+------------------------+--------------+
| TResNet-L        | 345                            | 1390                            | 316                    | 81.4         |
+------------------+--------------------------------+---------------------------------+------------------------+--------------+
| TResNet-XL       | 250                            | 1060                            | 240                    | **82.0**     |
+------------------+--------------------------------+---------------------------------+------------------------+--------------+

Comparison To Other Networks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Comparison of ResNet50 to top modern networks, with similar top-1
ImageNet accuracy. All measurements were done on Nvidia V100 GPU with
mixed precision. For gaining optimal speeds, training and inference were
measured on 90% of maximal possible batch size. Except TResNet-M, all
the models' ImageNet scores were taken from the `public
repository <https://github.com/rwightman/pytorch-image-models>`__, which
specialized in providing top implementations for modern networks. Except
EfficientNet-B1, which has input resolution of 240, all other models
have input resolution of 224.

+------------------+--------------------------------+---------------------------------+--------------+------------+
| Model            | Top Training Speed (img/sec)   | Top Inference Speed (img/sec)   | Top-1 Acc.   | Flops[G]   |
+==================+================================+=================================+==============+============+
| ResNet50         | **805**                        | 2830                            | 79.0         | 4.1        |
+------------------+--------------------------------+---------------------------------+--------------+------------+
| ResNet50-D       | 600                            | 2670                            | 79.3         | 4.4        |
+------------------+--------------------------------+---------------------------------+--------------+------------+
| ResNeXt50        | 490                            | 1940                            | 78.5         | 4.3        |
+------------------+--------------------------------+---------------------------------+--------------+------------+
| EfficientNetB1   | 440                            | 2740                            | 79.2         | 0.6        |
+------------------+--------------------------------+---------------------------------+--------------+------------+
| SEResNeXt50      | 400                            | 1770                            | 79.0         | 4.3        |
+------------------+--------------------------------+---------------------------------+--------------+------------+
| MixNet-L         | 400                            | 1400                            | 79.0         | 0.5        |
+------------------+--------------------------------+---------------------------------+--------------+------------+
| TResNet-M        | 730                            | **2930**                        | **80.7**     | 5.5        |
+------------------+--------------------------------+---------------------------------+--------------+------------+

Transfer Learning SotA Results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Comparison of TResNet to state-of-the-art models on transfer learning
datasets (only ImageNet-based transfer learning results). Models
inference speed is measured on a mixed precision V100 GPU. Since no
official implementation of Gpipe was provided, its inference speed is
unknown.

+------------------+-------------------+--------------+-----------------+---------+
| Dataset          | Model             | Top-1 Acc.   | Speed img/sec   | Input   |
+==================+===================+==============+=================+=========+
| CIFAR-10         | Gpipe             | **99.0**     | -               | 480     |
+------------------+-------------------+--------------+-----------------+---------+
| CIFAR-10         | TResNet-XL        | **99.0**     | **1060**        | 224     |
+------------------+-------------------+--------------+-----------------+---------+
| CIFAR-100        | EfficientNet-B7   | **91.7**     | 70              | 600     |
+------------------+-------------------+--------------+-----------------+---------+
| CIFAR-100        | TResNet-XL        | 91.5         | **1060**        | 224     |
+------------------+-------------------+--------------+-----------------+---------+
| Stanford Cars    | EfficientNet-B7   | 94.7         | 70              | 600     |
+------------------+-------------------+--------------+-----------------+---------+
| Stanford Cars    | TResNet-L         | **96.0**     | **500**         | 368     |
+------------------+-------------------+--------------+-----------------+---------+
| Oxford-Flowers   | EfficientNet-B7   | 98.8         | 70              | 600     |
+------------------+-------------------+--------------+-----------------+---------+
| Oxford-Flowers   | TResNet-L         | **99.1**     | **500**         | 368     |
+------------------+-------------------+--------------+-----------------+---------+

