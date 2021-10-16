# Project Instruction

## Installation
### 1. Install conda
### 2. Create an environment
    
    conda create -n your_env_name python=3.7
### 3. Activate the environment
    conda activate your_env_name
### 4. Install Packages
#### 4.1 Install Torch
##### 4.1.1 GPU support
    conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
##### 4.1.2 CPU 
    conda install pytorch torchvision torchaudio cpuonly -c pytorch
#### 4.2 Install Other necessary packages
    pip install -r project_root/requirements.txt

### 5. Command
#### 5.1 Preprocessing
    python project_root/manage.py --preprocessing --input "path/to/file.csv"
##### 5.2 Train
    python project_root/manage.py --train --input "path/to/processed_file.csv" --col_name "Predict Column name" --time_step 10 --epochs 500 

#### 5.3 Test
    python project_root/manage.py --model "path/to/model.pth" --input "path/to/processed_file.csv" --time_step 10 --col_name Close


## System Specificity

| Device      | Model |
| ----------- | ----------- |
| GPU       | Nvidia 1650 4G Geforce|
| CPU   | Core i5 - 4900f|
| RAM   | 16 G|
| OS   | Ubuntu 18.04 LTS|
| Cuda   | 11.2|
| GPU-Driver   | 460|






    