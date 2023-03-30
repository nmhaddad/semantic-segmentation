# Semantic segmentation of off-road images using transfer learning and DeepLabv3+
A transfer learning follow-up using DeepLabV3+ and the Yamaha-CMU Off-Road Dataset

### Author: Nate Haddad - *nhaddad2112[at]gmail[dot]com*

![Semantic segmentation of off-road images](media/seg_overlay.png)

## Abstract:
Many state-of-the-art deep learning algorithms require both a large training dataset and compute power, which for a variety of reasons is not always available to the user. Training large networks from scratch becomes tedious, and time consumer for the end-user. Transfer learning is a machine learning method where the transfer of knowledge from one domain to another, which ultimately eliminates the need to train from scratch using randomly initialized networks and benchmark datasets. Transfer is a popular technique in the computer vision field, given the vast compute and time resources required to train neural network architectures. In this paper, we will examine the effect of transfer learning on large encoder-decoder style deep neural networks applied to the task of semantic segmentation. DeepLabv3+ is one such architecture that was state-of-the-art at the time of its publishing in 2018. By applying depth-wise separable convolution to altrous spatial pyramid pooling and decoder modules, DeepLabv3+ was able to successfully incorporate techniques from the 2016 Xception model, resulting in faster, more powerful, and larger networks. We propose extending the use of a pre-trained DeepLabv3+ model to the challenging task of off-road perception. Utilizing the newly available Yamaha-CMU Off-Road Dataset, we successfully employ transfer learning techniques to a pre-trained model for the task of semantic segmentation of off-road images.  

Please email me for the full report

![ResNet101 inference on off-road images](media/resnet101_output.png)

## Installation:

### Setting up the Repository: 

The following instructions will set up the repository for training and inference. It creates a `data` directory, which is used to store training data. Steps 3 and 4 will download and unzip the *[Yamaha-CMU Off-Road Dataset](https://theairlab.org/yamaha-offroad-dataset/)* to the `data` directory.

1. `git clone https://github.com/nmhaddad/semantic-segmentation.git`
2. `mkdir data && cd data`
3. `wget https://cmu.box.com/s/3fngoljhcwhqf2z5cbepufh331qtesxt`
4. `unzip yamaha_v0.zip`

### Installing Dependencies with Anaconda
The simpliest way to get this repo up and running is to use our anaconda `environment.yaml` file:

`conda create -f environment.yaml`

To activate, run `conda activate semantic-segmentation`.

## Models

[Pretrained Models](https://drive.google.com/drive/folders/1Gmk8vOF9qBNMg3-TEL-st6KWieB4Af5e?usp=sharing)

## Running:

You can configure your training environment with the `config/config.yaml` file. This YAML file includes a variety of hyperparameters and paths for training and testing models. It's meant to simplify experimentation.

There are two notebooks, `training_demo.ipynb` and `inference_demo.ipynb` that make it simple to run the code here in the repository. Just make sure you have downloaded the *[Yamaha-CMU Off-Road Dataset](https://theairlab.org/yamaha-offroad-dataset/)* before training.

Run `python train.py` to run a standalone training session.

Run `python test.py` to run a standalone inference session on single images.

Run `python process_video.py` to run a standalone inference session on videos.

## References:

[1] Chen, Liang-Chieh, Zhu, Yukun, Papandreou, George, Schroff, Florian, and Adam, Hartwig. "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation.” Computer Vision – ECCV2018 (2018): 833-51. Web.  

[2] Chollet, Francois. "Xception: Deep Learning with Depthwise Separable Convolutions.” 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2017): 1800-807. Web.  

[3]  Daniel Maturana and Po-Wei Chou and Masashi Uenoyama and Sebastian Scherer, “Real-time Semantic Mapping for Autonomous Off-Road Navigation” in Maturana-2017-102768, September 2017, pp. 335 - 350.  

[4]  Stevo. Bozinovski  and  Ante  Fulgosi  (1976).  "The  influence of pattern similarity and transfer learning upon the training of a base perceptronB2.” (original in  Croatian) Proceedings of Symposium Informatica 3-121-5, Bled.  

[5] Stevo Bozinovski (2020) "Reminder of the first paper on transfer learning in neural networks, 1976”. Informatica 44: 291–302.  

[6] Pan, S.J.; Yang, Q. A survey on transfer learning. IEEE Trans. Knowl. Data Eng. 2010, 22, 1345–1359  

[7] M. S. Minhas, “Transfer Learning for Semantic Segmentation using PyTorch DeepLabv3,” GitHub.com/msminhas93, 12-Sep-2019. [Online]. Available: https://github.com/msminhas93/DeepLabv3FineTuning.
