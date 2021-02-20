# dcase2021_task2_baseline_ae
Autoencoder-based baseline system for DCASE2021 Challenge Task 2.

## Description
This system consists of two main scripts:
- `00_train.py`
  - "Development" mode: 
    - This script trains a model for each machine type by using the directory `dev_data/<machine_type>/train/`.
  - "Evaluation" mode: 
    - This script trains a model for each machine type by using the directory `eval_data/<machine_type>/train/`. (This directory will be from the "additional training dataset".)
- `01_test.py`
  - "Development" mode:
    - This script makes a csv file for each section including the anomaly scores for each wav file in the directories `dev_data/<machine_type>/source_test/` and `dev_data/<machine_type>/target_test/`.
    - The csv files are stored in the directory `result/`.
    - It also makes a csv file including AUC, pAUC, precision, recall, and F1-score for each section.
  - "Evaluation" mode: 
    - This script makes a csv file for each section including the anomaly scores for each wav file in the directories `eval_data/<machine_type>/source_test/` and `eval_data/<machine_type>/target_test/`. (These directories will be from the "evaluation dataset".)
    - The csv files are stored in the directory `result/`.

## Usage

### 1. Clone repository
Clone this repository from Github.

### 2. Download datasets
We will launch the datasets in three stages. 
So, please download the datasets in each stage:
- "Development dataset"
  - Download `dev_data_<machine_type>.zip` from https://zenodo.org/record/xxxxxxx.
- "Additional training dataset", i.e. the evaluation dataset for training
  - After launch, download `eval_data_train_<machine_type>.zip` from https://zenodo.org/record/yyyyyyy (not available until April. 1, 2021).
- "Evaluation dataset", i.e. the evaluation for test
  - After launch, download `eval_data_test_<machine_type>.zip` from https://zenodo.org/record/zzzzzzz (not available until June. 1, 2021).

### 3. Unzip dataset
Unzip the downloaded files and make the directory structure as follows:
- /dcase2021_task2_baseline_ae
    - /00_train.py
    - /01_test.py
    - /common.py
    - /keras_model.py
    - /baseline.yaml
    - /readme.md
- /dev_data
    - /fan
        - /train (Normal data in the **source** and **target** domains for all sections are included.)
            - /section_00_source_train_normal_0000_<attribute>.wav
            - ...
            - /section_00_source_train_normal_0999_<attribute>.wav
            - /section_00_target_train_normal_0000_<attribute>.wav
            - /section_00_target_train_normal_0001_<attribute>.wav
            - /section_00_target_train_normal_0002_<attribute>.wav
            - /section_01_source_train_normal_0000_<attribute>.wav
            - ...
            - /section_02_target_train_normal_0002_<attribute>.wav
        - /source_test (Normal and anomaly data in the **source** domain for all sections are included.)
            - /section_00_source_test_normal_0000.wav
            - ...
            - /section_00_source_test_normal_0099.wav
            - /section_00_source_test_anomaly_0000.wav
            - ...
            - /section_00_source_test_anomaly_0099.wav
            - /section_01_source_test_normal_0000.wav
            - ...
            - /section_02_source_test_anomaly_0099.wav
        - /target_test (Normal and anomaly data in the **target** domain for all sections are included.)
            - /section_00_target_test_normal_0000.wav
            - ...
            - /section_00_target_test_normal_0099.wav
            - /section_00_target_test_anomaly_0000.wav
            - ...
            - /section_00_target_test_anomaly_0099.wav
            - /section_01_target_test_normal_0000.wav
            - ...
            - /section_02_target_test_anomaly_0099.wav
    - /gearbox (The other machine types have the same directory structure as fan.)
    - /pump
    - /slider
    - /valve
    - /ToyCar
    - /ToyTrain
- /eval_data (Add this directory after launch)
    - /fan
        - /train (Unzipped "additional training dataset". Normal data in the **source** and **target** domains for all sections are included.)
            - /section_03_source_train_normal_0000_<attribute>.wav
            - ...
            - /section_03_source_train_normal_0999_<attribute>.wav
            - /section_03_target_train_normal_0000_<attribute>.wav
            - /section_03_target_train_normal_0001_<attribute>.wav
            - /section_03_target_train_normal_0002_<attribute>.wav
            - /section_04_source_train_normal_0000_<attribute>.wav
            - ...
            - /section_05_target_train_normal_0002_<attribute>.wav
        - /source_test (Unzipped "evaluation dataset". Normal and anomaly data in the **source** domain for all sections are included.)
            - /section_03_source_test_0000.wav
            - ...
            - /section_03_source_test_0199.wav
            - /section_04_source_test_0000.wav
            - ...
            - /section_05_source_test_0199.wav
        - /target_test (Unzipped "evaluation dataset". Normal and anomaly data in the **target** domain for all sections are included.)
            - /section_03_target_test_0000.wav
            - ...
            - /section_03_target_test_0199.wav
            - /section_04_target_test_0000.wav
            - ...
            - /section_05_target_test_0199.wav
    - /gearbox (The other machine types have the same directory structure as fan.)
    - /pump
    - /slider
    - /valve
    - /ToyCar
    - /ToyTrain

### 4. Change parameters
You can change parameters for feature extraction and model definition by editing `baseline.yaml`.

### 5. Run training script (for development dataset)
Run the training script `00_train.py`. 
Use the option `-d` for the development dataset `dev_data/<machine_type>/train/`.
```
$ python3.6 00_train.py -d
```
Options:

| Argument                    |                                   | Description                                                  | 
| --------------------------- | --------------------------------- | ------------------------------------------------------------ | 
| `-h`                        | `--help`                          | Application help.                                            | 
| `-v`                        | `--version`                       | Show application version.                                    | 
| `-d`                        | `--dev`                           | Mode for "development"                                       |  
| `-e`                        | `--eval`                          | Mode for "evaluation"                                        | 

`00_train.py` trains a model for each machine type and store the trained models in the directory `model/`.

### 6. Run test script (for development dataset)
Run the test script `01_test.py`.
Use the option `-d` for the development dataset **dev_data/<machine_type>/test/**.
```
$ python3.6 01_test.py -d
```
The options for `01_test.py` are the same as those for `00_train.py`.
`01_test.py` calculates an anomaly score for each wav file in the directories `dev_data/<machine_type>/source_test/` and `dev_data/<machine_type>/target_test/`.
A csv file for each section including the anomaly scores will be stored in the directory `result/`.
If the mode is "development", the script also outputs another csv file including AUC, pAUC, precision, recall, and F1-score for each section.

### 7. Check results
You can check the anomaly scores in the csv files `anomaly_score_<machine_type>_section_<section_index>_<domain>_test.csv` in the directory `result/`.
Each anomaly score corresponds to a wav file in the directories `dev_data/<machine_type>/source_test/` and `dev_data/<machine_type>/target_test/`:

`anomaly_score_fan_section_00_source_test.csv`
```
section_00_source_test_normal_0000.wav	-1.4423707
section_00_source_test_normal_0001.wav	-5.8763485
section_00_source_test_normal_0002.wav	-2.5255458
section_00_source_test_normal_0003.wav	-2.3934057
section_00_source_test_normal_0004.wav	-1.2815342
section_00_source_test_normal_0005.wav	-8.897109
  ...
```

Also, anomaly detection results after thresholding can be checked in the csv files `anomaly_score_<machine_type>_section_<section_index>_<domain>_test.csv`:

`decision_result_fan_section_00_source_test.csv`
```
section_00_source_test_normal_0000.wav	0
section_00_source_test_normal_0001.wav	1
section_00_source_test_normal_0002.wav	0
section_00_source_test_normal_0003.wav	0
section_00_source_test_normal_0004.wav	0
section_00_source_test_normal_0005.wav	1
  ...
```

Also, you can check performance indicators such as AUC, pAUC, precision, recall, and F1 score:

`result.csv`
```  
fan						
section	domain	AUC	pAUC	precision	recall	F1 score
0	source	0.75	0.533157895	0.684782609	0.63	0.65625
1	source	0.7988	0.591578947	0.795454545	0.35	0.486111111
2	source	0.7434	0.526842105	0.70212766	0.33	0.448979592
0	target	0.7816	0.638947368	0.74	0.74	0.74
1	target	0.5526	0.53	0.590361446	0.49	0.535519126
2	target	0.7786	0.653157895	0.510526316	0.97	0.668965517
arithmetic mean		0.734166667	0.578947368	0.670542096	0.585	0.589304224
harmonic mean		0.722561403	0.574327542	0.656046403	0.50429309	0.570246328
  ...
valve						
section	domain	AUC	pAUC	precision	recall	F1 score
0	source	0.5154	0.498947368	0.515151515	0.17	0.255639098
1	source	0.4625	0.495789474	0.416666667	0.1	0.161290323
2	source	0.5733	0.531052632	0.565217391	0.26	0.356164384
0	target	0.5313	0.504210526	0.657894737	0.25	0.362318841
1	target	0.6106	0.513684211	0.566037736	0.9	0.694980695
2	target	0.4998	0.511052632	0.551724138	0.32	0.405063291
arithmetic mean		0.53215	0.509122807	0.545448697	0.333333333	0.372576105
harmonic mean		0.527825198	0.508861719	0.535256741	0.214556838	0.306324124
						
		AUC	pAUC	precision	recall	F1 score
arithmetic mean over all machine types, sections, and domains		0.715579087	0.57952875	0.642966272	0.547498505	0.517271698
harmonic mean over all machine types, sections, and domains		0.689876899	0.573595787	0.61590263	0.306473307	0.409285863
```

### 8. Run training script for "additional training dataset" (after April 1, 2021)
After the "additional training dataset" is launched, download and unzip it.
Move it to `eval_data/<machine_type>/train/`.
Run the training script `00_train.py` with the option `-e`. 
```
$ python3.6 00_train.py -e
```
Models are trained by using the "additional training dataset" `eval_data/<machine_type>/train/`.

### 9. Run test script for "evaluation dataset" (after June 1, 2021)
After the "evaluation dataset" for test is launched, download and unzip it.
Move it to `eval_data/<machine_type>/source_test/` and `eval_data/<machine_type>/target_test/`.
Run the test script `01_test.py` with the option `-e`. 
```
$ python3.6 01_test.py -e
```
Anomaly scores are calculated using the "evaluation dataset", i.e., `eval_data/<machine_type>/source_test/` and `eval_data/<machine_type>/target_test/`.
The anomaly scores are stored as csv files in the directory `result/`.
You can submit the csv files for the challenge.
From the submitted csv files, we will calculate AUC, pAUC, and your ranking.

## Dependency
We develop the source code on Ubuntu 16.04 LTS and 18.04 LTS.
In addition, we checked performing on **Ubuntu 16.04 LTS**, **18.04 LTS**, **Cent OS 7**, and **Windows 10**.

### Software packages
- p7zip-full
- Python == 3.6.5
- FFmpeg

### Python packages
- Keras                         == 2.3.0
- Keras-Applications            == 1.0.8
- Keras-Preprocessing           == 1.0.5
- matplotlib                    == 3.0.3
- numpy                         == 1.16.0
- PyYAML                        == 5.1
- scikit-learn                  == 0.20.2
- librosa                       == 0.6.0
- audioread                     == 2.1.5 (more)
- setuptools                    == 41.0.0
- tensorflow                    == 1.15.0
- tqdm                          == 4.23.4
