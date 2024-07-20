# Script to setup the python virtual environment
# required for running the object detection script

# Clone the Mobility Challenge git repository
git clone https://github.com/kirtisanghi/Mobility_Challange.git

# conda must be installed and present in the PATH
# create a python virtual environment using conda
conda env create --file Mobility_Challange/mobility-challenge-custom-env.yml

# Activate the python virtual env
conda activate mobility-challenge-custom-env

# NOTE: This is required as we are
# using our custom fine-tuned model
# Clone the ultralytics repo
git clone https://github.com/ultralytics/ultralytics

# shellcheck disable=SC2164
cd ultralytics

git reset --hard 2071776a3672eb835d7c56cfff22114707765ac

wget https://gist.githubusercontent.com/Y-T-G/8f4fc0b78a0a559a06fe84ae4f359e6e/raw/17b1407fefeac86d089c4cf14f174c8bb44948af/add_head.patch

git apply add_head.patch

pip install -e .

cd ../

# shellcheck disable=SC2164
cd Mobility_Challange/object-detection/

python app.py Stn_HD_1.mp4 Output_Turning_Patterns.csv app.py

