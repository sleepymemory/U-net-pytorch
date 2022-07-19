# U-net on pytorch

Standard U-net implementation based on pytorch

### Tutorial

1.Install libraries

**windows**

[pytorch](https://pytorch.org/get-started/previous-versions/)

**cuda and  cudnn version**

[cuda 11.3](https://developer.nvidia.com/cuda-11.3.0-download-archive)

[cudnn v8.2.1](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.2.1.32/11.3_06072021/cudnn-11.3-windows-x64-v8.2.1.32.zip)

**pytorch install:**

 ``pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113``

**Standard libraries**

``pip install numpy``

``pip install Pillow``

``pip install tqdm``

### Data process

In the **json_files** folder of the **labelme_dataset** folder, place the json files marked with labelme.

run data_process.py

Raw image will generate in **raw** folder 

Mask image will generate in **mask** folder 

### Train model

Change the raw_image_path and mask_image_path as your dataset

Change the weight_path to where you want to save the model params

run train.py

### Use model 