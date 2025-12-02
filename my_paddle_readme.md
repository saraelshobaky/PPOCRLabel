# Creating the Paddle OCR training environment
-----------------------------------------------------------------
Please get rid and delete all old environemnts
Based on https://github.com/PFCCLab/PPOCRLabel/blob/main/README.md

## Building Environment (Train and Label)
```
$ conda create -n paddle-label-train-env python==3.13
$ conda activate paddle-label-train-env

$ pip install pillow requests safetensors typing-extensions packaging urllib3 jinja2

# If CPU
$ pip install paddlepaddle==3.2.2 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# If GPU   (For Legion and HPC cuda 13.0 hopefully HPC also compatible with cuda 13.0)
$ python -m pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu130/

$ pip install "paddleocr[all]" --no-cache-dir
$ pip install ppocrlabel --no-cache-dir

# For the labeling interface windowing on xcb (GNOME)
$ pip install pyqt5
```

### To solve the error of chart recognition
` $ vim ~/miniconda3/envs/pocrlabel3-env/lib/python3.13/site-packages/paddlex/inference/pipelines/layout_parsing/pipeline_v2.py `

Add an if statement @ line 212  around the chart recognition config and model block

```
        if self.use_chart_recognition:
            chart_recognition_config = config.get("SubModules", {}).get(
                "ChartRecognition",
                {"model_config_error": "config error for block_region_detection_model!"},
            )
            self.chart_recognition_model = self.create_model(
                chart_recognition_config,
            )
```

## To Annotate Training data

### Cloning source code
```
$ cd [PARENT_DIR]/

# Orinigal git source
$ git clone https://github.com/PFCCLab/PPOCRLabel.git

# Forked to own account, current working version
$ git clone https://github.com/saraelshobaky/PPOCRLabel.git
```

### Running the labeling App
```
$ cd [PARENT_DIR]/PPOCRLabel/

$ QT_QPA_PLATFORM=xcb \
LD_LIBRARY_PATH=$CONDA_PREFIX/lib \
QT_PLUGIN_PATH=$CONDA_PREFIX/lib/qt5/plugins \
python PPOCRLabel.py --lang en \
    --rec_model_dir  ~/.paddlex/official_models/arabic_PP-OCRv5_mobile_rec/  \
    --rec_model_name arabic_PP-OCRv5_mobile_rec \
    --det_model_dir ~/.paddlex/official_models/PP-OCRv5_server_det/   \
    --det_model_name PP-OCRv5_server_det


<!-- # On Work PC (Wayland)
$ conda activate pocrlabel3-env
$ QT_QPA_PLATFORM=wayland  python PPOCRLabel.py ..... -->


```

**\* Note that I adjusted the code of ppocrlabel and commited my version in the devpos, github**


## To Generate Training data
After labeling your data images, you have to export your trainind data in Paddle specific format with the structure and percentation of train,validate and test. 
i.e. $ python gen_ocr_train_val_test.py --trainValTestRatio  6:2:2 --datasetRootPath ../train_data

```
$ cd [PARENT_DIR]/PPOCRLabel/
$ conda activate paddle-label-train-env

$ python gen_ocr_train_val_test.py --trainValTestRatio 7:3:0 --datasetRootPath ~/Desktop/Paddle_training/a/  --convertArabicDigits true   --flipRTL true

==> output are in [PARENT_DIR]/train_data/rec/

==> --convertArabicDigits true  # Ensures to convert all english digits to Arabic Hindi digits (Default=true)

==> --flipRTL true  # Ensures to flip line characters (Except numbers, digits, decimals) to be read from left to right (special case of paddle ocr image scanning from left to write. This case not available in tesseract) (Default=true)

```

## Model Training
https://www.paddleocr.ai/main/en/version3.x/module_usage/text_recognition.html#4-secondary-development
In order to fine tune an already exisiting model with your labeled training data generated above


### Create the environment
```
$ cd [PARENT_DIR]/

# clone source code
$ git clone https://github.com/PaddlePaddle/PaddleOCR.git
# note that I didn't create a local environment for this repo as I didn't change it, so load it from source but ensure that you are using the latest version (using tags)


# activate already created paddle environment before 
$ conda activate paddle-label-train-env

# Note that  on the workPC, the environment is named 'paddle-ocr-env'

# install remaining libraries
$ cd [PARENT_DIR]/PaddleOCR/
$ pip install -r requirements.txt
```

###  Modify the Arabic yaml configuration file
Note: Check this files from git repo as I update it reguliarly
```
$ vim [PARENT_DIR]/PaddleOCR/configs/rec/PP-OCRv5/multi_language/arabic_PP-OCRv5_mobile_rec.yaml
```

```
  use_gpu: true 
  
  max_text_length: &max_text_length 120

  d2s_train_image_shape: [3, 48, 1280]
  d2s_train_image_shape: [3, 48, 1280]
  
  image_shape: [48, 1280, 3]
  scales: [[1280, 32], [1280, 48], [1280, 64]]  
  image_shape: [3, 48, 1280]
```


### Finetuning the model
```
$ cd [PARENT_DIR]/PaddleOCR/
$ conda activate paddle-label-train-env
```

#### Adjust the config file
```
$ vi configs/rec/PP-OCRv5/multi_language/arabic_PP-OCRv5_mobile_rec.yaml
```
-- ensure that you downloaded the arabic_PP-OCRv5_mobile_rec_pretrained.pdparams in the pretrained model path, as the pdiparams file is not re-trainable. Also ensure that the path of the config parameter 'pretrained_model' is absolute without '~' 
-- Adjust the 'label_file_list' param twice in the file

##### For small GPU memory
- You can decrease the batch size 2x or 4x but you have to decrease the learning rate also with same amount 2x or 4x. 
- Also you may decrease the number of workers
- You can also skip the data augmentation as it may not be need in text documents, as it fits the scenes only.
- What we did currently is to used an Amplifier only and didn't go through the above comments yet 
```
  use_visualdl: true #false  # <<<< CAHNGED 
  vdl_log_dir: ./output/vdl_log/   # <<<  ADDED (Where the graphs go)
  # --- AMP SETTINGS ADDED (To save memory) ---
  use_amp: true                    # <<< ADDED
  scale_loss: 1024.0               # <<< ADDED
  use_dynamic_loss_scaling: true   # <<< ADDED
```

#### Train the model
```
$ python tools/train.py -c configs/rec/PP-OCRv5/multi_language/arabic_PP-OCRv5_mobile_rec.yaml
```








### Export the model 
```
$ python tools/export_model.py \
    -c configs/rec/PP-OCRv5/multi_language/arabic_PP-OCRv5_mobile_rec.yaml \
    -o Global.pretrained_model=[PARENT_DIR]/PaddleOCR/output/arabic_rec_ppocr_v5/latest.pdparams \
    Global.save_inference_dir=[PARENT_DIR]/PaddleOCR/output/arabic_rec_ppocr_v5/tunedX/

```

--> Generated Tuned model in the following path: 
    [PARENT_DIR]/PaddleOCR/output/arabic_rec_ppocr_v5/tunedX/


### Run the labeling app with the tuned model above
```
####################Tuned Model
QT_QPA_PLATFORM=xcb \
LD_LIBRARY_PATH=$CONDA_PREFIX/lib \
QT_PLUGIN_PATH=$CONDA_PREFIX/lib/qt5/plugins \
python PPOCRLabel.py --lang en \
    --rec_model_dir  [PARENT_DIR]/PaddleOCR/output/arabic_rec_ppocr_v5/tunedX/  \
    --rec_model_name arabic_PP-OCRv5_mobile_rec \
    --det_model_dir ~/.paddlex/official_models/PP-OCRv5_server_det/   \
    --det_model_name PP-OCRv5_server_det 


####################Original Model, just change the rec_model_dir
QT_QPA_PLATFORM=xcb \
LD_LIBRARY_PATH=$CONDA_PREFIX/lib \
QT_PLUGIN_PATH=$CONDA_PREFIX/lib/qt5/plugins \
python PPOCRLabel.py --lang en \
    --rec_model_dir  ~/.paddlex/official_models/arabic_PP-OCRv5_mobile_rec/  \
    --rec_model_name arabic_PP-OCRv5_mobile_rec \
    --det_model_dir ~/.paddlex/official_models/PP-OCRv5_server_det/   \
    --det_model_name PP-OCRv5_server_det 
```


## GPU Training options (TOCHECK)

# Single-GPU training (default training method)
```
$ python3 tools/train.py -c configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml \
   -o Global.pretrained_model=./PP-OCRv5_server_rec_pretrained.pdparams
```

# Multi-GPU training, specify GPU IDs via the --gpus parameter
```
$ python3 -m paddle.distributed.launch --gpus '0,1,2,3'  tools/train.py -c configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml \
        -o Global.pretrained_model=./PP-OCRv5_server_rec_pretrained.pdparams
```





