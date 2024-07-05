# THE BENGALURU MOBILITY CHALLENGE, 2024

Bangalore mobility challenge is a hackathon organised by Bengaluru Traffic Police with collaboration with IISc and Centre for Data for Public Good. It aims creating innovative solutions to the traffic management problem in Bengaluru.

The hackathon will have two phases. 

1. The first phase will be about short-term traffic volume prediction, given video feeds from cameras installed at junctions.
2. The second phase will be about vehicle re-identification.


As part of this challenge we are proposing a solution to detect the different vehicles at the traffic junction from the CCTV feed using deep learning algorithms and use the obtained data for volume prediction.

### Phase 1.1 Object Detection and Tracking

*_Ultralytics YOLOv8 Models_* will be used for vehicle detection using their `yolov8` model with `track` task

#### Python Environment creation

We are using `conda` to manage the python virtual environment. We can use different virtual environments when using pre-trained model vs fine-tuned model. `mobility-challenge-env.yml` consists the config for creating a conda env for running inference against pre-trained model and `mobility-challenge-custom-env.yml` consists the config for creating conda env for running inference against fine-tuned model

1. Create a conda env using the config yaml
```shell
conda env create --file mobility-challenge-env.yml
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

### Phase 1.2 Vehicle Volume Prediction


### Phase 2.0 Vehicle Re-identification