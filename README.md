# dsa4266-dsa4266666-genomics
This repository is a project done by **dsa4266666** team for the module DSA4266 Sense-making Case Analysis: Science and Technology in National University of Singapore.

## Table of Contents

- [Usage](#Usage)
  - [Connect AWS Machine](#Connect-AWS-Machine)
  - [Install Resources](#Install-Resources)
- [Task 1](#Task-1)
  - [Download Datasets](#Download-Datasets)
  - [Training Model (Optional)](#Training-Model-(Optional))
  - [Generate Predictions](#Generate-Predictions)
- [Task 2](#Task-2)
  - [Download Datasets from AWS](#Download-Datasets-from-AWS)
  - [Preprocess Datasets](#Preprocess-Datasets)
  - [Feature Engineering](#Feature-Engineering)
  - [Generate Final Predictions](#Generate-Final-Predictions)
- [Contributors](#Contributors)
- [Citations](#Citations)
- [Software License](#Software-License)

## Usage
### Connect AWS Machine
There are two ways of connecting to the AWS machine.
- Connect to the machine via the Research Gateway portal <br/>
    1. Select #citations
    2. Click on the `SSH/RDP` button.
    3. Insert 'Ubuntu' under `Username` section.
    4. Click on `Choose a file` button and select your pem file
    5. Click on `Submit` button.

- Connect to the machine via ssh on your local laptop <br/>
    1. Select the machine to be used.
    2. Click on the `Outputs` button.
    3. Check your `InstanceDNSName`. (eg. `ec2-13-250-105-39.ap-southeast-1.compute.amazonaws.com`)
    4. Open your local terminal/powershell.
    5. Use the follwing command: <br/>
    `ssh -i '/path/to/filename.pem' ubuntu@<InstanceDNSName>`
        - For MacOS: <br/>
        `ssh -i /path/to/filename.pem ubuntu@ec2-13-250-105-39.ap-southeast-1.compute.amazonaws.com`
        - For Windows: <br/>
        `ssh -i \path\to\filename.pem ubuntu@ec2-13-250-105-39.ap-southeast-1.compute.amazonaws.com`

        (Note: for MacOS users, you may need to run `chmod 400 /path/to/filename.pem` once to resolve the unprotected private key file error.)

### Install Resources
First, change to home directory.
```
cd ~
```
To install the resources, use `git clone` to clone the GitHub repository into your machine.
```
git clone https://github.com/Guan-Yee/dsa4266-dsa4266666-genomics.git
cd dsa4266-dsa4266666-genomics
```
Next, install pip and the required packages using the following code:
```
sudo apt install python3-pip
pip install -r requirements.txt
```

## Task 1
Write a computational method that predicts m6A RNA modification from direct RNA-Seq data. The method should be able to train a new new model, and make predictions on unseen test data.

To predict m6A with test datasets, you may run through [Download Datasets](#Download-Datasets) and [Generate Predictions](#Generate-Predictions) sections.

### Download Datasets
First, change your directory to `task_1`.
```
cd task_1
```
If you are not at `dsa4266-dsa4266666-genomics` directory, use:
```
cd ~/dsa4266-dsa4266666-genomics/task_2
```
Then, download the datasets using the following code:
```
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=1SDLz3enY7TyaTNxTs7bXcKAAwJfU67M-" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=1SDLz3enY7TyaTNxTs7bXcKAAwJfU67M-" -o dataset0.json.gz

curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=1i55Ty8RUZc11Dz3GgnifA41_c-GrXePi" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=1i55Ty8RUZc11Dz3GgnifA41_c-GrXePi" -o dataset1.json.gz

curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=14eexLMcm-bYtVwgtpwbJTlw-Zh_7dXXK" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=14eexLMcm-bYtVwgtpwbJTlw-Zh_7dXXK" -o dataset2.json.gz

curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=1AqeWnV5SOyXN9-Y-9Rku2BQ5uiShe9NP" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=1AqeWnV5SOyXN9-Y-9Rku2BQ5uiShe9NP" -o dataset3.json.gz

curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=1JNpp7uaFXJpaoLwnRVVqj8MGf4aB2Zs0" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=1JNpp7uaFXJpaoLwnRVVqj8MGf4aB2Zs0" -o data.info
```
After running the code, the repository will contain the following files:
- `1_Task_1_Model_Training_Script.py`
- `2_Task_1_Prediction.py`
- `LGBM_tuned.pkl`
- `dataset0.json.gz`
- `dataset1.json.gz`
- `dataset2.json.gz`
- `dataset3.json.gz`
- `data.info`

### Training Model (Optional)
The LightGBM model `LGBM_tuned.pkl` is already available in this repository.

If you want to train the model, you may remove the previous model and train it again by running these code:
```
rm LGBM_tuned.pkl
python3 1_Task_1_Model_Training_Script.py
```
At the end, a new LightGBM model `LGBM_tuned.pkl` will be generated.

### Generate Predictions
To generate the predictions for the datasets with the LightGBM model, you may run `2_Task_1_Prediction.py` using the following code:

(Note: Please ensure that `dataset0.json.gz`, `dataset1.json.gz`, `dataset2.json.gz`, and `dataset3.json.gz` are available in the repository.)

```
python3 2_Task_1_Prediction.py
```

At the end, 4 CSV files containing the predictions will be generated:
- `dataset0_predicted_m6A_site.csv`
- `dataset1_predicted_m6A_site.csv`
- `dataset2_predicted_m6A_site.csv`
- `dataset3_predicted_m6A_site.csv`

You may check the first few lines of the CSV file using `head`.

For example, you may check the first 5 lines of `dataset0_predicted_m6A_site.csv` by running
```
head -n 5 dataset0_predicted_m6A_site.csv
```
The output should look similar to below:
```
transcript_id,transcript_position,score
ENST00000000233,244,0.031690000153040805
ENST00000000233,261,0.20029018278348798
ENST00000000233,316,0.04074177045204055
ENST00000000233,332,0.117464322262655
```

## Task 2
Predict m6A RNA modifications in all samples from the SG-NEx data using your method. Describe the results and compare them across the different cell lines. Summarise and visualise your observations.

All data is available as .json files through the SG-NEx project: https://github.com/GoekeLab/sg-nex-data which is hosted on AWS (https://registry.opendata.aws/sgnex/).

### Download Datasets from AWS
First, change your directory to `task_1`.
```
cd task_2
```
If you are not at `dsa4266-dsa4266666-genomics` directory, use:
```
cd ~/dsa4266-dsa4266666-genomics/task_2
```
Create the `data`, `preprocessed_data`, `final_data`, and `lgbm_prediction` repositories to store the data.
```
mkdir data
mkdir preprocessed_data
mkdir final_data
mkdir lgbm_prediction
```
Then, download the datasets from AWS:
```
aws s3 sync --no-sign-request s3://sg-nex-data/data/processed_data/m6Anet/ data
```
After downloading the datasets successfully, there will have 12 folders in `data` repository:
- `SGNex_A549_directRNA_replicate5_run1`
- `SGNex_Hct116_directRNA_replicate4_run3`
- `SGNex_K562_directRNA_replicate5_run1`
- `SGNex_A549_directRNA_replicate6_run1`
- `SGNex_HepG2_directRNA_replicate5_run2`
- `SGNex_K562_directRNA_replicate6_run1`
- `SGNex_Hct116_directRNA_replicate3_run1`
- `SGNex_HepG2_directRNA_replicate6_run1`
- `SGNex_MCF7_directRNA_replicate3_run1`
- `SGNex_Hct116_directRNA_replicate3_run4`
- `SGNex_K562_directRNA_replicate4_run1`
- `SGNex_MCF7_directRNA_replicate4_run1`

You may list out the folders in `data` repository using `ls`.
```
ls data
```

### Preprocess Datasets
Running `1_Task_2_Preprocessing.py` will preprocess the datasets and generate preprocessed datasets in `preprocessed_data` repository.
```
python3 1_Task_2_Preprocessing.py
```
After preprocessing, `preprocessed_data` will contain 12 Parquet files:
- `SGNex_A549_directRNA_replicate5_run1_intermediate.parquet`
- `SGNex_A549_directRNA_replicate6_run1_intermediate.parquet`
- `SGNex_Hct116_directRNA_replicate3_run1_intermediate.parquet`
- `SGNex_Hct116_directRNA_replicate3_run4_intermediate.parquet`
- `SGNex_Hct116_directRNA_replicate4_run3_intermediate.parquet`
- `SGNex_HepG2_directRNA_replicate5_run2_intermediate.parquet`
- `SGNex_HepG2_directRNA_replicate6_run1_intermediate.parquet`
- `SGNex_K562_directRNA_replicate4_run1_intermediate.parquet`
- `SGNex_K562_directRNA_replicate5_run1_intermediate.parquet`
- `SGNex_K562_directRNA_replicate6_run1_intermediate.parquet`
- `SGNex_MCF7_directRNA_replicate3_run1_intermediate.parquet`
- `SGNex_MCF7_directRNA_replicate4_run1_intermediate.parquet`

You may list out the folders in `preprocessed_data` repository using `ls`.
```
ls preprocessed_data
```

### Feature Engineering
To generate the final datasets used for prediction, we need to do feature engineering on the current datasets.

Use the following code to generate the final datasets:

(Note: It may take few hours to process all datasets.)
```
python3 2_Task_2_Feature_Engineering.py
```
At the end, 12 final datasets will be generated in `final_data` repository:
- `SGNex_A549_directRNA_replicate5_run1_final.parquet`
- `SGNex_A549_directRNA_replicate6_run1_final.parquet`
- `SGNex_Hct116_directRNA_replicate3_run1_final.parquet`
- `SGNex_Hct116_directRNA_replicate3_run4_final.parquet`
- `SGNex_Hct116_directRNA_replicate4_run3_final.parquet`
- `SGNex_HepG2_directRNA_replicate5_run2_final.parquet`
- `SGNex_HepG2_directRNA_replicate6_run1_final.parquet`
- `SGNex_K562_directRNA_replicate4_run1_final.parquet`
- `SGNex_K562_directRNA_replicate5_run1_final.parquet`
- `SGNex_K562_directRNA_replicate6_run1_final.parquet`
- `SGNex_MCF7_directRNA_replicate3_run1_final.parquet`
- `SGNex_MCF7_directRNA_replicate4_run1_final.parquet`

You may list out the folders in `final_data` repository using `ls`.
```
ls final_data
```

### Generate Final Predictions
To generate the predictions for the datasets with the LightGBM model, you may run `3_Task_2_Predict_LGBM.py` using the following code:

(Note: Please ensure that `LGBM_tuned.pkl` is available in `../task_1/` repositry and the required final datasets are available in `final_data/` repository)
```
python3 3_Task_2_Predict_LGBM.py
```
At the end, 12 CSV files containing the predictions will be generated:
- `SGNex_A549_directRNA_replicate5_run1_predict_lgbm.csv`
- `SGNex_A549_directRNA_replicate6_run1_predict_lgbm.csv`
- `SGNex_Hct116_directRNA_replicate3_run1_predict_lgbm.csv`
- `SGNex_Hct116_directRNA_replicate3_run4_predict_lgbm.csv`
- `SGNex_Hct116_directRNA_replicate4_run3_predict_lgbm.csv`
- `SGNex_HepG2_directRNA_replicate5_run2_predict_lgbm.csv`
- `SGNex_HepG2_directRNA_replicate6_run1_predict_lgbm.csv`
- `SGNex_K562_directRNA_replicate4_run1_predict_lgbm.csv`
- `SGNex_K562_directRNA_replicate5_run1_predict_lgbm.csv`
- `SGNex_K562_directRNA_replicate6_run1_predict_lgbm.csv`
- `SGNex_MCF7_directRNA_replicate3_run1_predict_lgbm.csv`
- `SGNex_MCF7_directRNA_replicate4_run1_predict_lgbm.csv`

You may check the first few lines of the CSV file using `head`.

For example, you may check the first 5 lines of `SGNex_A549_directRNA_replicate5_run1_predict_lgbm.csv` by running
```
head -n 5 lgbm_prediction/SGNex_A549_directRNA_replicate5_run1_predict_lgbm.csv
```
The output should look similar to below:
```
transcript_id,transcript_position,score
ENST00000000233,244,0.07827733772961283
ENST00000000233,261,0.1637502051452505
ENST00000000233,316,0.2275218085451278
ENST00000000233,332,0.25732804015417066
```

## Contributors
- Chin Chun Yuan
- Chua Ming Feng
- Daryl Wang Yan
- Loo Guan Yee
- Rena Chong Pei Qi

## Citations
- Chen, Ying, et al. "A systematic benchmark of Nanopore long read RNA sequencing for transcript level analysis in human cell lines." bioRxiv (2021). doi: https://doi.org/10.1101/2021.04.21.440736
- Hendra, C., Pratanwanich, P.N., Wan, Y.K. et al. Detection of m6A from direct RNA sequencing using a multiple instance learning framework. Nat Methods 19, 1590–1598 (2022). https://doi.org/10.1038/s41592-022-01666-1
- The pandas development team. (2023). pandas-dev/pandas: Pandas (v2.0.3). Zenodo. https://doi.org/10.5281/zenodo.8092754
- Pedregosa, F. et al., 2011. Scikit-learn: Machine learning in Python. Journal of machine learning research, 12(Oct), pp.2825–2830.
- Ke, G. et al., 2017. Lightgbm: A highly efficient gradient boosting decision tree. Advances in neural information processing systems, 30, pp.3146–3154.
- Richardson N, Cook I, Crane N, Dunnington D, Francois R, Keane J, Moldovan-Grunfeld D, Ooms J, Wujciak-Jens J, Apache Arrow (2023). arrow: Integration to 'Apache' 'Arrow'. https://github.com/apache/arrow/, https://arrow.apache.org/docs/r/.

## Software License
This project is licensed under the [MIT License](LICENSE).