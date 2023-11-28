# LDRM: Degradation Rectify Model for Low-light Imaging via Color-Monochrome Cameras <br> (Accepted by ACM MM'23)
# Abstract:

Low-light imaging task aims to approximate low-light scenes as perceived by human eyes. Existing methods usually pursue higher brightness, resulting in unrealistic exposure. Inspired by Human Vision System (HVS), where rods perceive more lights while cones perceive more colors, we propose a Low-light Degradation Rectify Model (LDRM) with color-monochrome cameras to solve this problem. First, we propose to use a low-ISO color camera and a high-ISO monochrome camera for low-light imaging under short-exposure of less than 0.1s. Short-exposure could avoid motion blurriness, while monochrome camera captures more photons than color camera. By mimicing HVS, this capture system could benefit low-light imaging. Second, we propose an LDRM model to fuse the color-monochrome image pair into a high-quality image. In this model, we separately restore UV and Y channels through chrominance and luminance branches and use monochrome image to guide the restoration of luminance. We also propose a latent code embedding method to improve the restorations of both branches. Third, we create a Low-light Color-Monochrome benchmark (LCM), including both synthetic and real-world datasets, to examine low-light imaging quality of LDRM and the state-of-the-art methods. Experimental results demonstrate the superior performance of LDRM with visually pleasing results.

[[Paper Download]]([LDRM: Degradation Rectify Model for Low-light Imaging via Color-Monochrome Cameras | Proceedings of the 31st ACM International Conference on Multimedia](https://dl.acm.org/doi/abs/10.1145/3581783.3613792))

You can also refer our works on other low-level vision applications!

[LMQFormer: A Laplace-Prior-Guided Mask Query Transformer for Lightweight Snow Removal | IEEE TCSVT]([ieeexplore.ieee.org/abstract/document/10092769](https://ieeexplore.ieee.org/abstract/document/10092769))

# Network Architecture

<img src=".\img\Method.png" alt="Method" style="zoom:50%;" />

![Network](.\img\Network.png)

# LCM Benchmark and Results

We create a Low-light Color-Monochrome Benchmark (LCM) that contains both synthetic and real-world datasets.

It can be downloaded from：

Link: https://pan.baidu.com/s/1GE_xID7QDDc7Hn1EK4dKzw 

Extract code: LDRM

![Syn](.\img\Syn.png)

Qualitative results on Synthetic dataset of LCM. Please zoom in for better visual quality.



![Real](.\img\Real.png)

Qualitative results on Real-world dataset of LCM. Please zoom in for better visual quality.

# Setup and environment

#### To generate the recovered result you need:

1. Python 3.7
2. CPU or NVIDIA GPU + CUDA CuDNN
3. Pytorch 1.8.0
4. python-opencv

#### Testing

Please replace weights_dir data_dir and result_dir in test.py, and put your testset in data_dir.

The pre-trained model can be found in checkpoint folder.

#### Pre-trained model

It can be downloaded from：

Link: https://pan.baidu.com/s/1GE_xID7QDDc7Hn1EK4dKzw 

Extract code: LDRM


# Citations
Please cite this paper in your publications if it is helpful for your tasks:    

Bibtex:
```
@inproceedings{10.1145/3581783.3613792,
author = {Lin, Junhong and Pei, Shufan and Chen, Bing and Jiang, Nanfeng and Gao, Wei and Zhao, Tiesong},
title = {LDRM: Degradation Rectify Model for Low-Light Imaging via Color-Monochrome Cameras},
year = {2023},
isbn = {9798400701085},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3581783.3613792},
doi = {10.1145/3581783.3613792},
booktitle = {Proceedings of the 31st ACM International Conference on Multimedia},
pages = {8406–8414},
numpages = {9},
keywords = {human vision system, low-light imaging, color-monochrome cameras},
location = {<conf-loc>, <city>Ottawa ON</city>, <country>Canada</country>, </conf-loc>},
series = {MM '23}
}

```
