# VRID

VRID is a Real-time video streaming re-id tool

## requirements

* python == 3.8
* opencv == 4.5.5.64
* numpy == 1.22.3
* tensorflow-gpu == 2.8.0
* pytorch == 1.9.0+cull
* torchvision == 0.10.0+cull
* pandas == 1.4.1
* scikit-learn == 1.0.2


## General usage

The re-id process is divided into the following steps.

1. Pedestrian Detection and Filtering

You can run the _**detect.py**_ file in the **_video2img_** directory to complete the pedestrian detection step

2. Portrait foreground separation.

The portrait foreground can be separated by running the _**seg.py**_ file in the _**preprocess**_ directory. Note that the portrait foreground format saved after separation is **.png**.

3. Color feature map generation and Pedestrian upper body crop.

You can run the _**caculate_color_num.py**_ file in the _**preprocess**_ folder to generate color feature maps for the corresponding datasets. Note that if the recognizability of the input image is too low, the corresponding color feature map may not be generated.

4. Feature extraction.

Just run the _**get_features.py**_ to get features.Both color feature maps and pedestrian images use this file for feature extraction

5. Re-Identification and Evaluation.

Finally, run **_retrieval_mAP.py_** to get the re-id result, such as the following two examples:

Use color features and original image features
```
python3 retrieval_mAP.py --data_name div_seg_all_person --color_data_name div_color_seg_all_person --if_concat True
```
Only use original image features
```
python3 retrieval_mAP.py --data_name div_seg_all_person
```
## Dataset 

We only provide the features of the pedestrian data used in the paper, which can be downloaded from ***,and placed in the _**results/features**_ directory