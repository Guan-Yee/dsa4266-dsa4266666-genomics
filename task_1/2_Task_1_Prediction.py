#!/usr/bin/env python
# coding: utf-8

# In[13]:


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
# Line 91 - Aggregation Feature Engineering
# Line 169 - Binary Columns Feature Engineering
# Line 246 - Feature Engineering Master function
# Line 333 - Feature Engineered Dataset Preprocessing
# Line 355 - LightGBM Initialisation and Prediction
# Line 385 - Prediction Master Function

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
########## Preliminary Feature Preprocessing for Test Dataset###################

# This function will parse the dataset0.json.gz file
# Pre-requisite: The test json.gz file must be found in ..Dataset/ directory
# Reture a list containing dictionary entries
def generate_all_data(test_data_json_path):
    # Initialise an empty list
    all_data = []

    # Start parsing the json gzip file
    with gzip.open(test_data_json_path, 'rt') as file:
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

# This is the master function that return a full-preprocessed dataframe
# Pre-requisite: test dataset.json.gz must be present in the project storage folder
# Return a fully preprocessed pandas dataframe
def preprocessTestDataset(test_data_json_path):
    all_data = generate_all_data(test_data_json_path)
    df = generate_dataframe(all_data)
    print("Preliminary Feature Preprocessing Completed!")
    return df

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

  # Iterate through the columns from index 3 to 11
  for i in range(3, 12):
      cols[i] = cols[i] + '_read_{}'.format(criteria)

  # Update the column names in the DataFrame
  grouped_df1.columns = cols

  # Return the grouped_df1 dataframe
  return grouped_df1

def generateAggMeanDf(df_copy, agg_columns):
    grouped_df1 = df_copy.groupby(['transcript_id',
                                   'transcript_position',
                                   'Nucleotide Sequence'])[agg_columns].mean().reset_index()

    grouped_df1 = rename12Cols(grouped_df1, "mean")
    return grouped_df1

def generateAggMedianDf(df_copy, agg_columns):
    grouped_df2 = df_copy.groupby(['transcript_id',
                                    'transcript_position',
                                   'Nucleotide Sequence'])[agg_columns].median().reset_index()

    grouped_df2 = rename12Cols(grouped_df2, "median")
    return grouped_df2


def generateAggMinDf(df_copy, agg_columns):
    grouped_df3 = df_copy.groupby(['transcript_id',
                                   'transcript_position',
                                   'Nucleotide Sequence'])[agg_columns].min().reset_index()

    grouped_df3 = rename12Cols(grouped_df3, "min")
    return grouped_df3

def generateAggMaxDf(df_copy, agg_columns):
    grouped_df4 = df_copy.groupby(['transcript_id',
                                   'transcript_position',
                                   'Nucleotide Sequence'])[agg_columns].max().reset_index()

    grouped_df4 = rename12Cols(grouped_df4, "max")
    return grouped_df4

def generateAggStdDf(df_copy, agg_columns):
    grouped_df5 = df_copy.groupby(['transcript_id',
                                   'transcript_position',
                                   'Nucleotide Sequence'])[agg_columns].std(ddof=0).reset_index()

    grouped_df5 = rename12Cols(grouped_df5, "std")
    return grouped_df5

def generateAggLastDf(df_copy, agg_columns):
    grouped_df6 = df_copy.groupby(['transcript_id',
                                   'transcript_position',
                                   'Nucleotide Sequence'])[agg_columns].last().reset_index()

    grouped_df6 = rename12Cols(grouped_df6, "last")
    return grouped_df6

def generateAggCountDf(df_copy, agg_columns):
    grouped_df7 = df_copy.groupby(['transcript_id',
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
################## Feature Engineering Master Function #########################

def generateAggDf(original_data):
    # Create a deep copy of the input DataFrame
    df_copy = original_data.copy()

    # Define the columns to aggregate
    agg_columns = df_copy.columns[3:12]

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
    merged_df = pd.merge(grouped_df1, grouped_df2, on=['transcript_id', 'transcript_position',
                                                       'Nucleotide Sequence'])
    merged_df = pd.merge(merged_df, grouped_df3, on=['transcript_id', 'transcript_position',
                                                     'Nucleotide Sequence'])
    merged_df = pd.merge(merged_df, grouped_df4, on=['transcript_id', 'transcript_position',
                                                     'Nucleotide Sequence'])
    merged_df = pd.merge(merged_df, grouped_df5, on=['transcript_id', 'transcript_position',
                                                     'Nucleotide Sequence'])
    merged_df = pd.merge(merged_df, grouped_df6, on=['transcript_id', 'transcript_position',
                                                     'Nucleotide Sequence'])
    merged_df = pd.merge(merged_df, grouped_df7, on=['transcript_id', 'transcript_position',
                                                     'Nucleotide Sequence'])

    # Nucleotide Seq cols
    final_df = createNucleotideSeqCols(merged_df)

    return final_df

def normalizeColumns(df):
    # Extract the columns to be normalized (columns 2 to 56)
    columns_to_normalize = df.columns[2:57]

    # Create a StandardScaler instance
    scaler = StandardScaler()

    # Normalize the selected columns
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

    return df

def shortlistUsefulFeatures(df):
    cols = ['transcript_id', 'transcript_position',
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
              'Last_Nucleotide_T']
    df1 = df[cols]
    return df1

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################## Feature Engineered Dataset Preprocessing ####################

def preprocess_data(df):
    # List of column names to convert to numeric
    columns = list(df.columns)

    # Remove the gene_id, transcript_id, Nucleotide_Sequence and label
    cols = [col for col in columns if col not in ['gene_id','transcript_id',
                                                  'Nucleotide Sequence',
                                                  'label']]
    # Shortlist these numeric columns
    X = df[cols]

    # Return X, y
    return X

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################### LightGBM Initialisation and Prediction #####################

def initialiseLGBM(model_path):
    lgbm_model = pickle.load(open(model_path, 'rb'))
    print("LightGBM Model initialised!")
    return lgbm_model

def predictCellLine(model, df, filename):
    X = preprocess_data(df)
    y_pred = model.predict_proba(X)[:, 1] # Probability score
    df1 = df.copy()
    df1['score'] = y_pred
    
    # Begin to shortlist the important columns for the dataset
    cols_for_submission = ['transcript_id', 'transcript_position', 'score']
    df2 = df1[cols_for_submission]
    print("Dataset Prediction Completed!")
    
    # Specify the export directory
    destination_directory = os.path.join(os.getcwd(), filename)
    
    # Export to the respective directory
    df2.to_csv(destination_directory, index = False)

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
######################### Prediction Master Function ###########################

def generatePrediction(test_data_json_path, model_path, filename):
    df = preprocessTestDataset(test_data_json_path)
    df1 = generateAggDf(df)
    df2 = normalizeColumns(df1)
    df3 = shortlistUsefulFeatures(df2)
    model = initialiseLGBM(model_path)
    predictCellLine(model, df3, filename)

test_data_json_path = os.path.join(os.getcwd(), "dataset0.json.gz") # Change to aws project storage and test dataset
model_path = os.path.join(os.getcwd(), "LGBM_tuned.pkl") # Change to aws project storage
filename = "dataset0_predicted_m6A_site.csv"
generatePrediction(test_data_json_path, model_path, filename)

test_data_json_path = os.path.join(os.getcwd(), "dataset1.json.gz") # Change to aws project storage and test dataset
model_path = os.path.join(os.getcwd(), "LGBM_tuned.pkl") # Change to aws project storage
filename = "dataset1_predicted_m6A_site.csv"
generatePrediction(test_data_json_path, model_path, filename)

test_data_json_path = os.path.join(os.getcwd(), "dataset2.json.gz") # Change to aws project storage and test dataset
model_path = os.path.join(os.getcwd(), "LGBM_tuned.pkl") # Change to aws project storage
filename = "dataset2_predicted_m6A_site.csv"
generatePrediction(test_data_json_path, model_path, filename)

test_data_json_path = os.path.join(os.getcwd(), "dataset3.json.gz") # Change to aws project storage and test dataset
model_path = os.path.join(os.getcwd(), "LGBM_tuned.pkl") # Change to aws project storage
filename = "dataset3_predicted_m6A_site.csv"
generatePrediction(test_data_json_path, model_path, filename)

