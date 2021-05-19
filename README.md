# Semantic segmentation of off-road images using transfer learning and DeepLabv3+
A transfer learning follow-up using DeepLabV3+ and the Yamaha-CMU Off-Road Dataset

### Authors:
Nate Haddad - *haddad[dot]na[at]northeastern[dot]edu*  
Amit Mulay - *mulay[dot]am[at]northeastern[dot]edu*  

Northeastern University - *academic project*

![Semantic segmentation of off-road images](media/seg_overlay.png)

## Abstract:
Many state-of-the-art deep learning algorithms require both a large training dataset and compute power, which for a variety of reasons is not always available to the user. Training large networks from scratch becomes tedious, and time consumer for the end-user. Transfer learning is a machine learning method where the transfer of knowledge from one domain to another, which ultimately eliminates the need to train from scratch using randomly initialized networks and benchmark datasets. Transfer is a popular technique in the computer vision field, given the vast compute and time resources required to train neural network architectures. In this paper, we will examine the effect of transfer learning on large encoder-decoder style deep neural networks applied to the task of semantic segmentation. DeepLabv3+ is one such architecture that was state-of-the-art at the time of its publishing in 2018. By applying depth-wise separable convolution to altrous spatial pyramid pooling and decoder modules, DeepLabv3+ was able to successfully incorporate techniques from the 2016 Xception model, resulting in faster, more powerful, and larger networks. We propose extending the use of a pre-trained DeepLabv3+ model to the challenging task of off-road perception. Utilizing the newly available Yamaha-CMU Off-Road Dataset, we successfully employ transfer learning techniques to a pre-trained model for the task of semantic segmentation of off-road images.  

[Read the report](media/report.pdf)

## Installation:

### Required Dependencies:
- python 3.8 (other versions may work as well, but untested)
- pytorch
- torchvision
- opencv (python)
- matplotlib
- tqdm

### Setting up the Repository: 

The following instructions will set up the repository for training and inference. It creates two required directories, `models` which is used to store saved models, and `data` which is used to store training data. Steps 4 and 5 will download and unzip the *[Yamaha-CMU Off-Road Dataset](https://theairlab.org/yamaha-offroad-dataset/)* to the `data` directory.

1. `git clone <repo>`
2. `mkdir models`
3. `mkdir data && cd data`
4. `wget https://cmu.box.com/s/3fngoljhcwhqf2z5cbepufh331qtesxt`
5. `unzip yamaha_v0.zip`
6. Use the `requirements.txt` file to set up your environment.

## Running:

There are two notebooks, `training_demo.ipynb` and `inference_demo.ipynb` that make it simple to run the code here in the repository. Just make sure you have downloaded the *[Yamaha-CMU Off-Road Dataset](https://theairlab.org/yamaha-offroad-dataset/)* before training. Unless you have downloaded a model from this repo or somewhere else, it's likely you will need to train the model on your own. We intend to put a model up, but currently there isn't one.

You can also run `python3 train.py` to run a standalone training session. We have not yet set up an argparser for this, and you will need to adjust hyperparameters accordingly.

## References:

[1] Chen, Liang-Chieh, Zhu, Yukun, Papandreou, George, Schroff, Florian, and Adam, Hartwig. "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation.” Computer Vision – ECCV2018 (2018): 833-51. Web.  

[2] Chollet, Francois. "Xception: Deep Learning with Depthwise Separable Convolutions.” 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2017): 1800-807. Web.  

[3]  Daniel Maturana and Po-Wei Chou and Masashi Uenoyama and Sebastian Scherer, “Real-time Semantic Mapping for Autonomous Off-Road Navigation” in Maturana-2017-102768, September 2017, pp. 335 - 350.  

[4]  Stevo. Bozinovski  and  Ante  Fulgosi  (1976).  "The  influence of pattern similarity and transfer learning upon the training of a base perceptronB2.” (original in  Croatian) Proceedings of Symposium Informatica 3-121-5, Bled.  

[5] Stevo Bozinovski (2020) "Reminder of the first paper on transfer learning in neural networks, 1976”. Informatica 44: 291–302.  

[6] Pan, S.J.; Yang, Q. A survey on transfer learning. IEEE Trans. Knowl. Data Eng. 2010, 22, 1345–1359  

[7] M. S. Minhas, “Transfer Learning for Semantic Segmentation using PyTorch DeepLabv3,” GitHub.com/msminhas93, 12-Sep-2019. [Online]. Available: https://github.com/msminhas93/DeepLabv3FineTuning.
