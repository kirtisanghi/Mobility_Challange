# THE BENGALURU MOBILITY CHALLENGE, 2024

Bangalore mobility challenge is a hackathon organised by Bengaluru Traffic Police with collaboration with IISc and Centre for Data for Public Good. It aims creating innovative solutions to the traffic management problem in Bengaluru.

The hackathon will have two phases. 

1. The first phase will be about short-term traffic volume prediction, given video feeds from cameras installed at junctions.
2. The second phase will be about vehicle re-identification.


As part of this challenge we are proposing a solution to detect the different vehicles at the traffic junction from the CCTV feed using deep learning algorithms and use the obtained data for volume prediction.

## Phase 1.1 Object Detection and Tracking

*_Ultralytics YOLOv8 Models_* will be used for vehicle detection using their `yolov8` model with `track` task

### Python Environment creation

We are using `conda` to manage the python virtual environment. We can use different virtual environments when using pre-trained model vs fine-tuned model. `python-env.yml` consists the config for creating conda env for running inference against fine-tuned model

1. Create a conda env using the config yaml
```shell
conda env create --file python-env.yml
```

2. Activate the conda env
```shell
conda activate
```

3. Deactivate the conda env
```shell
conda deactivate
```

**ultralytics** package needs to be patched to be able to work with fined tuned model. Below provided are the additional steps required to be performed on the env to be able to use it for inference.
1. Clone `ultralytics` git repo and change to the root directory of repo
```shell
git clone https://github.com/ultralytics/ultralytics
cd ultralytics
```

2. Reset the git head to `2071776a3672eb835d7c56cfff22114707765ac`
```shell
git reset --hard 2071776a3672eb835d7c56cfff22114707765ac
```

3. Download the patch to be applied
```shell
wget https://gist.githubusercontent.com/Y-T-G/8f4fc0b78a0a559a06fe84ae4f359e6e/raw/17b1407fefeac86d089c4cf14f174c8bb44948af/add_head.patch
```

4. Apply the patches
```shell
git apply add_head.patch
```

5. Build and install `ultralytics` python package
```shell
pip install -e .
```

### Execution of Object Detection Script
`ml` package contains all the code for object detection, inference and other utility functions. The objective of running `ml` package is to detect the number of vehicles going in different directions at any particular junction for which the cctv camera feed (15 min clip) is provided. The number of vehicles are detected based on their classes eg `Cars`, `Bicycle`, `Two-Wheeler`, `Three-Wheeler`, `LCV`, `Truck` and `Bus`

The object detection script can be executed in one of the following ways. Note that the script is executed from `object-detection` folder

#### Invoking `ml` package
The video clip is the input to script which outputs the detection results in the csv file. The script can be executed with video clip available in the local directory or in s3 bucket.
```shell
usage: main.py [-h] (--input-file-path INPUT_FILE_PATH | --s3-input-path-prefix S3_INPUT_PATH_PREFIX) --output-file-path OUTPUT_FILE_PATH [--push-to-gcs]

Object Detection and Tracking Script

options:
  -h, --help            show this help message and exit
  --input-file-path INPUT_FILE_PATH
                        Path to the video file stored in local
  --s3-input-path-prefix S3_INPUT_PATH_PREFIX
                        Prefix of the video file stored in ieee s3 bucket
  --output-file-path OUTPUT_FILE_PATH
                        csv file name for output of object detection
  --push-to-gcs         push the output files to pre-defined gcs bucket
```

* Invoking script having video clip locally. A flag `--input-path` needs to be passed followed by its value. Optionally, one can also pass the file path where the csv file should be generated using `--output-file-name` flag. The below example shows the execution of script for a video clip `HP_Ptrl_Bnk_BEL_Rd_FIX_2_time_2024-05-14T07:30:02_000.mp4` available in `videos` folder along with the output filename as `detection-results.csv` 
    ```shell
  python -m ml.main --input-file-path "videos/HP_Ptrl_Bnk_BEL_Rd_FIX_2_time_2024-05-14T07:30:02_000.mp4" --output-file-name "detection-results.csv"
  ```

* Invoking script having video clip stored in AWS cloud's s3 bucket. You can either pass the full path to the s3 file or the prefix for the file (in this case all the files with this script will be used for object detection) available in `ieee-dataport` s3 bucket. A _*signed url*_ with be generated for the s3 file and that will be used to feed to the object detection model. Optionally, the file path to generate the csv file can also be passed using --output-file-name flag.
  Note that the script used AWS `boto3` python sdk to make AWS API calls, and it uses an aws cli profile named `hackathon`. Hence, before running executing below command make sure you have an aws profile configured with name `hackathon` having enough permissions.
    ```shell
  python -m ml.main --s3-input-path-prefix "competition/1253487/13083/Videos/2024-05-14/Ayyappa_Temple_FIX_1_time_2024-05-14T07:30:02_000.mp4" --output-file-name "detection-results.csv"
  ```


#### invoking `app.py` package
Alternatively, the `ml` package can be invoked using `app.py` module, which internally calls the `ml` package. This is wrapper and specifically made available to be used while leaderboard/phase submission and evaluation as suggested in [Leaderboard submission guidelines.pdf](https://drive.google.com/file/d/15t8_dtQjTBx3ueIR_eVTx2PPQtwsRcsD/view)
```shell
python app.py
[warn]: required arguments missing, please provide all the required arguments and execute the script in following way

python app.py {input_file.json} {output_file.json}

input_file      -> {'Cam_ID': {'Vid_1': '/path_to_vid_1', 'Vid_2': '/path_to_vid_2'}}
output_file     -> {'Cam_ID': {'Cumulative Counts': counts_by_class_turning_pattern, 'Predicted Counts': counts_by_class_turning_pattern }}
```
Invoke the script for `Ayyappa_Temple_FIX_1_time_2024-05-14T07:30:02_000.mp4` video clip and output the result csv at `Ayyappa_Temple_FIX_1_time_2024-05-14T07:30:02_000.csv`
```shell
python app.py input_file.json output_file.json
```

## Phase 1.2 Vehicle Volume Prediction

We are using LSTM (long-short term memory) RNN to implement timeseries forecasting to predict the number of vehicles for next 30 minutes in all the possible turning patterns for the target vehicle class. We have two LSTM layer with input tensor having sequence size 15 , two dropout layers and a dense layer.
The time series data generated by the object detection model will be used to train the LST model and then run the prediction on the data to get the vehicle counts for next 30 minutes



## Phase 2.0 Vehicle Re-identification