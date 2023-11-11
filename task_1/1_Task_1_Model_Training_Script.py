#!/usr/bin/env python
# coding: utf-8

# In[26]:


import os
import gzip
import json
import pandas as pd
import itertools
import lightgbm as lgb
import pickle

# Normalisation library
from sklearn.preprocessing import StandardScaler

# Content Page
# Line 27 - Preliminary Feature Preprocessing
# Line 124 - Aggregation Feature Engineering
# Line 202 - Binary Columns Feature Engineering
# Line 279 - Feature Engineering Master function
# Line 363 - Feature Engineered Dataset Preprocessing
# Line 414 - Training LGBM model
# Line 433 - Task 1 Model Training Script

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################# Preliminary Feature Preprocessing ############################

# This function will intialise the data_info folder found in Dataset folder.
# Pre-requisite: The data_info file must be found in ..Dataset/ directory.
# Return data_info in pandas dataframe format.
def initialise_data_info(file_path1):
    # DSA4266/Dataset/data.info path
    # This is a csv file mentioned in the lecture.
    data_info = pd.read_csv(file_path1)

    # Return this data_info
    return data_info

# This function will parse the dataset0.json.gz file
# Pre-requisite: The dataset0.json.gz file must be found in ..Dataset/ directory
# Reture a list containing dictionary entries
def generate_all_data(file_path2):
    # Initialise an empty list
    all_data = []

    # Start parsing the json gzip file
    with gzip.open(file_path2, 'rt') as file:
        for line in file:
            try:
                # Parse each line as a JSON object and append it to the list
                data = json.loads(line)
                all_data.append(data)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON: {line}")
    # Return the list
    return all_data

# This function will convert the dict in the list present to pandas dataframe.
# all_data list must be generated from generate_all_data() function.
# Return a pandas dataframe.
def generate_dataframe(all_data):
    # Initialise an empty list
    flat_data = []

    # Iterate through the each json format in the list
    for entry in all_data:
        for transcript, values in entry.items():
            for pos, subdata in values.items():
                for sequence, values in subdata.items():
                    for val in values:
                        flat_data.append([transcript, pos, sequence, *val])

    # Create a Pandas DataFrame
    columns = ['transcript_id', 'transcript_position', 'Nucleotide Sequence',
            '-1 dwelling time', '-1 sd signal','-1 mean signal',
            'central dwelling time', 'central sd signal','central mean signal',
            '+1 dwelling time', '+1 sd signal','+1 mean signal']
    df = pd.DataFrame(flat_data, columns=columns)

    # Convert the transcript_position to int64 format
    df['transcript_position'] = df['transcript_position'].astype('int64')

    # Return the dataframe
    return df

# This function will convert the dict in the list present to pandas dataframe.
# Pre-requsite: df must be generated from generate_dataframe function and
# data_info must be generated from generate_all_data() function.
# Return a merged pandas dataframe.
def merge_dataframe(data_info, df):
    # Gene_id must be added for the group k-fold training error later
    subset_cols = ['gene_id','transcript_id', 'transcript_position', 'label']
    subset_data_info = data_info[subset_cols]
    # Perform left join on the dataset
    combined_df = pd.merge(df, subset_data_info,
                        on = ['transcript_id', 'transcript_position'],
                        how = 'left')
    # Shift the `gene_id` to the first column in the list
    column_order = ['gene_id'] + [col for col in combined_df.columns if col != 'gene_id']

    # Rearrange the columns again
    combined_df = combined_df[column_order]
    # Return combined_df
    return combined_df

# This is the master function that return a full-preprocessed dataframe
# Pre-requisite: data_info and dataset0.json.gz must be present in dataset folder
# Return a fully preprocessed pandas dataframe
def preprocess_dataset(data_json_path, data_info_path):
    all_data = generate_all_data(data_json_path)
    df = generate_dataframe(all_data)
    data_info = initialise_data_info(data_info_path)
    final_df = merge_dataframe(data_info, df)
    print("Preliminary Feature Preprocessing Completed!")
    return final_df

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################# Aggregation Feature Engineering ##############################

def rename12Cols(grouped_df1, criteria):
  # Get the column names from the DataFrame
  cols = list(grouped_df1.columns)

  # Iterate through the columns from index 4 to 12
  for i in range(4, 13):
      cols[i] = cols[i] + '_read_{}'.format(criteria)

  # Update the column names in the DataFrame
  grouped_df1.columns = cols

  # Return the grouped_df1 dataframe
  return grouped_df1

def generateAggMeanDf(df_copy, agg_columns):
    grouped_df1 = df_copy.groupby(['gene_id', 'transcript_id',
                                   'transcript_position',
                                   'Nucleotide Sequence'])[agg_columns].mean().reset_index()

    grouped_df1 = rename12Cols(grouped_df1, "mean")
    return grouped_df1

def generateAggMedianDf(df_copy, agg_columns):
    grouped_df2 = df_copy.groupby(['gene_id', 'transcript_id',
                                    'transcript_position',
                                   'Nucleotide Sequence'])[agg_columns].median().reset_index()

    grouped_df2 = rename12Cols(grouped_df2, "median")
    return grouped_df2


def generateAggMinDf(df_copy, agg_columns):
    grouped_df3 = df_copy.groupby(['gene_id', 'transcript_id',
                                   'transcript_position',
                                   'Nucleotide Sequence'])[agg_columns].min().reset_index()

    grouped_df3 = rename12Cols(grouped_df3, "min")
    return grouped_df3

def generateAggMaxDf(df_copy, agg_columns):
    grouped_df4 = df_copy.groupby(['gene_id', 'transcript_id',
                                   'transcript_position',
                                   'Nucleotide Sequence'])[agg_columns].max().reset_index()

    grouped_df4 = rename12Cols(grouped_df4, "max")
    return grouped_df4

def generateAggStdDf(df_copy, agg_columns):
    grouped_df5 = df_copy.groupby(['gene_id', 'transcript_id',
                                   'transcript_position',
                                   'Nucleotide Sequence'])[agg_columns].std(ddof=0).reset_index()

    grouped_df5 = rename12Cols(grouped_df5, "std")
    return grouped_df5

def generateAggLastDf(df_copy, agg_columns):
    grouped_df6 = df_copy.groupby(['gene_id', 'transcript_id',
                                   'transcript_position',
                                   'Nucleotide Sequence'])[agg_columns].last().reset_index()

    grouped_df6 = rename12Cols(grouped_df6, "last")
    return grouped_df6

def generateAggCountDf(df_copy, agg_columns):
    grouped_df7 = df_copy.groupby(['gene_id', 'transcript_id',
                                   'transcript_position',
                                   'Nucleotide Sequence']).size().reset_index(name='Read counts')

    return grouped_df7

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################# Binary Columns Feature Engineering ###########################

# This function will create 18 Motifs that fulfills the DRACH motif criterion.
# Pre-requsite: itertools must be imported in the first place
# Return a list of 18 sequences.
def createDRACHList():
    # Define the mapping of characters to nucleotides
    mapping = {
      'D': ['A', 'G', 'T'],
      'R': ['G', 'A'],
      'A': ['A'],
      'C': ['C'],
      'H': ['A', 'C', 'T']
    }

    # Generate all possible combinations using itertools.product
    combinations = itertools.product(*[mapping[char] for char in 'DRACH'])
    # Join the combinations into 5-letter sequences
    sequences = [''.join(seq) for seq in combinations]
    return sequences


def decomposeNucleotideSeq(df):
    df1 = df.copy()
    df1['Mid Nucleotide Sequence'] = df1['Nucleotide Sequence'].str[1:-1]
    df1['First Nucleotide Char'] = df1['Nucleotide Sequence'].str[0]
    df1['Last Nucleotide Char'] = df1['Nucleotide Sequence'].str[-1]
    return df1

# Getter function to create the 18 columns of
def addDRACHCols(DRACH_motifs, df1):
    df2 = df1.copy()
    for pattern in DRACH_motifs:
        # Create empty list to store binary numbers 1 or 0
        temp_list = []
        # Iterate through the dataframe
        for index, row in df1.iterrows():
            # Check if the keyword exists in the row['Nucleotide Sequence']
            if pattern in row['Mid Nucleotide Sequence']:
                temp_list.append(1)
            else:  # Cannot be found
                temp_list.append(0)
        # Create a new column based on the pattern
        df2['{}_motif'.format(pattern)] = temp_list
        print('{} completed!'.format(pattern))
    return df2

# Getter function to create 4 columns of first nucleotide and last nucleotide variation
def addFirstLastNucleotideCharCols(df2):
    df3 = df2.copy()
    # One-hot encode 'First Nucleotide Char'
    first_nucleotide_encoded = pd.get_dummies(df3['First Nucleotide Char'], prefix='First_Nucleotide')

    # One-hot encode 'Last Nucleotide Char'
    last_nucleotide_encoded = pd.get_dummies(df3['Last Nucleotide Char'], prefix='Last_Nucleotide')

    # Concatenate the one-hot encoded columns back to the original DataFrame
    df4 = pd.concat([df3, first_nucleotide_encoded, last_nucleotide_encoded], axis=1)

    # Drop the original 'First Nucleotide Char' and 'Last Nucleotide Char' columns
    df4 = df4.drop(['First Nucleotide Char', 'Last Nucleotide Char', 'Mid Nucleotide Sequence',
                  'Nucleotide Sequence'], axis=1)
    return df4

def createNucleotideSeqCols(df):
    DRACH_motifs = createDRACHList()
    df1 = decomposeNucleotideSeq(df)
    df2 = addDRACHCols(DRACH_motifs, df1)
    df3 = addFirstLastNucleotideCharCols(df2)
    return df3

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################# Feature Engineering Master function ##########################

def generateAggDf(original_data, data_info):
    # Create a deep copy of the input DataFrame
    df_copy = original_data.copy()

    # Define the columns to aggregate
    agg_columns = df_copy.columns[4:13]

    # Mean
    grouped_df1 = generateAggMeanDf(df_copy, agg_columns)
    print('finished mean agg')
    # Median
    grouped_df2 = generateAggMedianDf(df_copy, agg_columns)
    print('finished median agg')
    # Min
    grouped_df3 = generateAggMinDf(df_copy, agg_columns)
    print('finished min agg')
    # Max
    grouped_df4 = generateAggMaxDf(df_copy, agg_columns)
    print('finished max agg')
    # Std
    grouped_df5 = generateAggStdDf(df_copy, agg_columns)
    print('finished std agg')
    # Last
    grouped_df6 = generateAggLastDf(df_copy, agg_columns)
    print('finished agglast')
    # Read count
    grouped_df7 = generateAggCountDf(df_copy, agg_columns)
    print('finished count agg')

    # Merge the dataframes based on the specified criteria
    merged_df = pd.merge(grouped_df1, grouped_df2, on=['gene_id', 'transcript_id', 'transcript_position', 'Nucleotide Sequence'])
    merged_df = pd.merge(merged_df, grouped_df3, on=['gene_id', 'transcript_id', 'transcript_position', 'Nucleotide Sequence'])
    merged_df = pd.merge(merged_df, grouped_df4, on=['gene_id', 'transcript_id', 'transcript_position', 'Nucleotide Sequence'])
    merged_df = pd.merge(merged_df, grouped_df5, on=['gene_id', 'transcript_id', 'transcript_position', 'Nucleotide Sequence'])
    merged_df = pd.merge(merged_df, grouped_df6, on=['gene_id', 'transcript_id', 'transcript_position', 'Nucleotide Sequence'])
    merged_df = pd.merge(merged_df, grouped_df7, on=['gene_id', 'transcript_id', 'transcript_position', 'Nucleotide Sequence'])

    # Nucleotide Seq cols
    temp_df = createNucleotideSeqCols(merged_df)

    # Left join the label
    final_df = pd.merge(temp_df, data_info,  on=['gene_id', 'transcript_id', 'transcript_position'])
    return final_df


def normalizeColumns(df):
    # Extract the columns to be normalized (columns 3 to 57)
    columns_to_normalize = df.columns[3:58]

    # Create a StandardScaler instance
    scaler = StandardScaler()

    # Normalize the selected columns
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

    return df

def shortlistUsefulFeatures(df):
    cols_41 = ['transcript_id', 'transcript_position',
              'central mean signal_read_std', 'central dwelling time_read_last',
              '+1 sd signal_read_min', '-1 dwelling time_read_min',
              '-1 mean signal_read_std', '+1 dwelling time_read_last',
              'central sd signal_read_min', 'central dwelling time_read_min',
              '+1 sd signal_read_last', '+1 mean signal_read_std',
              '+1 dwelling time_read_min', '-1 dwelling time_read_last',
              'Read counts', 'AGACA_motif', 'AGACC_motif', 'AGACT_motif',
              'AAACA_motif', 'AAACC_motif', 'AAACT_motif', 'GGACA_motif',
              'GGACC_motif', 'GGACT_motif', 'GAACA_motif', 'GAACC_motif',
              'GAACT_motif', 'TGACA_motif', 'TGACC_motif', 'TGACT_motif',
              'TAACA_motif', 'TAACC_motif', 'TAACT_motif', 'First_Nucleotide_A',
              'First_Nucleotide_C', 'First_Nucleotide_G', 'First_Nucleotide_T',
              'Last_Nucleotide_A', 'Last_Nucleotide_C', 'Last_Nucleotide_G',
              'Last_Nucleotide_T','label']
    df1 = df[cols_41]
    return df1

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
############## Feature Engineered Dataset Preprocessing ########################

def preprocessTrainData(df):
    # List of column names to convert to numeric
    columns = list(df.columns)

    # Remove the gene_id, transcript_id, Nucleotide_Sequence and label
    cols = [col for col in columns if col not in ['gene_id',
                                                  'transcript_id',
                                                  'Nucleotide Sequence',
                                                  'label']]
    # Shortlist these numeric columns
    X = df[cols]
    # Label as the target variable
    y = df['label']

    # Return X, y
    return X, y

def trainLightGBM(X, y):

    # Initialize LightGBM parameters (you can customize these)
    params = {
      'metric': 'binary_logloss',
      'boosting_type': 'gbdt',
      'n_estimators': 1000,
      'learning_rate': 0.0011579523456318834,
      'num_leaves': 250,
      'subsample': 0.1158264372257786,
      'lambda_l1': 0.028884517875667533,
      'lambda_l2': 5.290019606898125e-07,
      'bagging_fraction': 0.38868086017497033,
      'bagging_freq': 6,
      'feature_fraction': 0.9013039159121444,
      'colsample_bytree': 0.19466124114211408,
      'min_data_in_leaf': 58,
      'scale_pos_weight': (y == 0).sum()/ (y == 1).sum(),
      'verbose': -1,
      'random_state': 42
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(X,y, categorical_feature=list(range(14, 40)))
    
    return model

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
############################# Training Model ###################################

def generateTrainedLGBM(data_json_path, data_info_path):
    df= preprocess_dataset(data_json_path, data_info_path)
    data_label = initialise_data_info(data_info_path)
    df1 = generateAggDf(df, data_label)
    df2 = normalizeColumns(df1)
    df3 = shortlistUsefulFeatures(df2)
    X, y = preprocessTrainData(df3)
    model = trainLightGBM(X, y)
    filename = "LGBM_tuned.pkl"
    # Specify the file path in the current directory
    filename = os.path.join(os.getcwd(), "studies/ProjectStorage/task_1/LGBM_tuned.pkl") # Change to aws project storage
    # Export the model to the current directory
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
        
data_json_path = os.path.join(os.getcwd(), "studies/ProjectStorage/task_1/dataset0.json.gz") # Change to aws project storage
data_info_path = os.path.join(os.getcwd(), "studies/ProjectStorage/task_1/data.info") # Change to aws project storage
generateTrainedLGBM(data_json_path, data_info_path)

