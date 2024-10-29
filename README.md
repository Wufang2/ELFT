# ELFT

## Installation

<details>
  <summary>Installation</summary>

   1. Clone this repo

   2. Install Pytorch and torchvision

   3. Install other needed packages
   ```sh
   pip install -r requirements.txt
   ```
</details>




## Dataset


1.RSOD: https://github.com/RSIA-LIESMARS-WHU/RSOD-Dataset-

2.NWPU-VHR-10: https://github.com/Gaoshuaikun/NWPU-VHR-10

3.PASCAL VOC2007: http://host.robots.ox.ac.uk/pascal/VOC/




## Train


1.This article uses the COCO format for training. Before training, please organize dataset as following:
```
COCODIR/
  ├── train2017/
  ├── val2017/
  └── annotations/
  	├── instances_train2017.json
  	└── instances_val2017.json
```

2.Tune two parameters in a config file:
- Tuning the `num_classes` to the number of classes to detect in the dataset.
- Tuning the parameter `dn_labebook_size` to ensure that `dn_labebook_size >= num_classes + 1`

3.Start network training. The commands below reproduce the results.

    coco_path=$1
    python main.py \
        --output_dir logs/R50 -c config/ELFT/ELFT_4scale.py --coco_path $coco_path \
        --options dn_scalar=100 embed_init_tgt=TRUE \
        dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
        dn_box_noise_scale=1.0


## Evalate


Perform the command below.
    
    coco_path=$1
    checkpoint=$2
    python main.py \
      --output_dir logs/R50-%j \
        -c config/ELFT/ELFT_4scale.py --coco_path $coco_path  \
        --eval --resume $checkpoint \
        --options dn_scalar=100 embed_init_tgt=TRUE \
        dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
        dn_box_noise_scale=1.0

## Inference and Visualizations

For inference and visualizations, we provide inference.py as an example.
<details>
   <summary>Inference</summary>

   1. Change the path of the model config file and model checkpoint

   2. Get an example and visualize it

   3. Run inference.py

</details>

