## Image-Text Retrieval using BLIP on Flickr30K Dataset




<img src="BLIP.gif" width="700">

This is the PyTorch code of the <a href="https://arxiv.org/abs/2201.12086">BLIP paper</a> [[blog](https://blog.salesforceairesearch.com/blip-bootstrapping-language-image-pretraining/)]. The code has been tested on PyTorch 2.0.0+cu117.
To create the conda environment with the updated dependencies. \
run <pre/>conda env create -f environment.yml </pre> 

To install the dependencies, run <pre/>pip install -r requirements.txt</pre> 




### Pre-trained checkpoints:
Num. pre-train images | BLIP w/ ViT-B | BLIP w/ ViT-B and CapFilt-L | BLIP w/ ViT-L 
--- | :---: | :---: | :---: 
14M | <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_14M.pth">Download</a>| - | -
129M | <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth">Download</a>| <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth">Download</a> | <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large.pth">Download</a>


### Image-Text Retrieval:
1. Download the Flickr30k dataset. 
I downloaded it from Kaggle using Kaggle API. 
Use the following command after setting KAggle credentials on your computer.
<pre/>kaggle datasets download adityajn105/flickr30k </pre> 
and set 'image_root' in configs/retrieval_flickr.yaml accordingly.

3. To evaluate the finetuned BLIP model on Flickr30k, run:
<pre>python -m torch.distributed.run --nproc_per_node=1 train_retrieval.py \
--config ./configs/retrieval_flickr.yaml \
--output_dir output/retrieval_flickr \
--evaluate</pre> 
3. To finetune the pre-trained checkpoint using 1 A100 GPUs, first set 'pretrained' in configs/retrieval_flickr.yaml as "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth". Then run:
<pre>python -m torch.distributed.run --nproc_per_node=1 train_retrieval.py \
--config ./configs/retrieval_flickr.yaml \
--output_dir output/retrieval_flickr_finetuned </pre> 

### Description of key files:
train_retrieval.py: Script for training image retrieval models.\
pretrain.py: Python script for pretraining models.\
requirements.txt: Lists dependencies for the project.\
train_vqa.py: Script for training on Visual Question Answering tasks.\
demo.ipynb: Jupyter notebook demo of the project.\
predict.py: Script for making predictions using trained models.\
train_nlvr.py: Script for training on Natural Language Visual Reasoning tasks.\
utils.py: Utility functions for the project.\
train_caption.py: Script for training on image captioning tasks.\
eval_retrieval_video.py: Evaluation script for video retrieval.\
eval_nocaps.py: Evaluation script for the nocaps dataset.

Subdirectories:\
output: Contains logs and outputs from various training and evaluation processes.\
  retrieval_flickr, retrieval_flickr_finetuned, retrieval_coco: Folders for storing outputs specific to image retrieval tasks on different datasets.\
annotation: Likely contains JSON files for annotations related to various datasets like Flickr30K and COCO.\
models: Contains Python scripts defining different models and their components.\
configs: Configuration files for different training and evaluation setups.\
  retrieval_flickr.yaml: All the hyperparameters can be defined here. The pre-trained BLIP model checkpoint is also mentioned here.\

data: Scripts and utilities for handling various datasets.\
transform: Contains scripts for data transformation and augmentation.



