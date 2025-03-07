# Manga Coloring Model

This model is currently in development and is not yet complete.  
Kaggle was used as the dataset for training: [Kaggle Dataset](https://www.kaggle.com/datasets/ultraamvking/colored-manga)  

The architecture is based on Pix2Pix, whose paper can be found here: [Pix2Pix Paper](https://arxiv.org/abs/1611.07004)  

## Prerequisites

- Installed Python  
- PyTorch library ([https://pytorch.org/](https://pytorch.org/))  
- Download the pretrained weights ([Google Drive Link](https://drive.google.com/file/d/1WPh5zppEproC_YDjSYPmkQ9Z50ikiwIK/view?usp=drive_link))  

## Installation

- Clone the repository  

## Inference

Simply place the black-and-white image in the folder, named `image.jpg` by default, and run the `infer.py` script.  
The output will be saved as `image_colorized.png`.  

## Training

To train the AI, you need to have WandB and pandas installed (Python libraries).  

By default, running `train.py` trains the AI from scratch and saves its weights at each epoch.  
If you want to resume training from pretrained weights, uncomment lines 69 and 70 in `train.py`.  
Training information will be displayed on WandB.  

### Dataset Setup

- Install the OpenCV library  
- Place the images in `dataset/rgb/`  
- Create a folder `dataset/bw/`  
- Run `utils/setup_dataset.py`  

The dataset is now ready to use.  

## Example Results

From left to right, we have:  
- The black-and-white image  
- The original colored image  
- The model-generated colored image  

![media_images_Validation Image_2100_8c707e38b91c65e0012c](https://github.com/Aul16/ColorationManga/assets/39156836/f9641e32-5cdd-4674-994a-f5c1498bc33c)  

## Limitations

Since the black-and-white dataset is created by an algorithm from the RGB dataset to form image pairs,  
this model struggles to generalize on arbitrary manga images.  
A new unsupervised learning model is being developed to address this issue.  
