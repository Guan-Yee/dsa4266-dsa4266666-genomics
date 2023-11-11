"""### Feature engineering"""

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import itertools

"""#### Functions for aggregation of numerical values"""

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
                                 'transcript_position', 'Nucleotide Sequence'])[agg_columns].mean().reset_index()

  grouped_df1 = rename12Cols(grouped_df1, "mean")
  return grouped_df1

def generateAggMedianDf(df_copy, agg_columns):
  grouped_df2 = df_copy.groupby(['transcript_id',
                                 'transcript_position', 'Nucleotide Sequence'])[agg_columns].median().reset_index()

  grouped_df2 = rename12Cols(grouped_df2, "median")
  return grouped_df2


def generateAggMinDf(df_copy, agg_columns):
  grouped_df3 = df_copy.groupby(['transcript_id',
                                 'transcript_position', 'Nucleotide Sequence'])[agg_columns].min().reset_index()

  grouped_df3 = rename12Cols(grouped_df3, "min")
  return grouped_df3

def generateAggMaxDf(df_copy, agg_columns):
  grouped_df4 = df_copy.groupby(['transcript_id',
                                 'transcript_position', 'Nucleotide Sequence'])[agg_columns].max().reset_index()

  grouped_df4 = rename12Cols(grouped_df4, "max")
  return grouped_df4

def generateAggStdDf(df_copy, agg_columns):
  grouped_df5 = df_copy.groupby(['transcript_id',
                                 'transcript_position', 'Nucleotide Sequence'])[agg_columns].std().reset_index()

  grouped_df5 = rename12Cols(grouped_df5, "std")
  return grouped_df5

def generateAggLastDf(df_copy, agg_columns):
  grouped_df6 = df_copy.groupby(['transcript_id',
                                 'transcript_position', 'Nucleotide Sequence'])[agg_columns].last().reset_index()

  grouped_df6 = rename12Cols(grouped_df6, "last")
  return grouped_df6

def generateAggSkewnessDf(df_copy, agg_columns):
  def calculate_skewness(column):
      return column.skew()

  grouped_df7 = df_copy.groupby(['transcript_id',
                                 'transcript_position', 'Nucleotide Sequence'])[agg_columns].agg(calculate_skewness).reset_index()

  grouped_df7 = rename12Cols(grouped_df7, "skewness")
  return grouped_df7

def generateAggKurtosisDf(df_copy, agg_columns):
  def calculate_kurtosis(column):
    return column.kurtosis()
  grouped_df8 = df_copy.groupby(['transcript_id',
                                 'transcript_position', 'Nucleotide Sequence'])[agg_columns].agg(calculate_kurtosis).reset_index()

  grouped_df8 = rename12Cols(grouped_df8, "kurtosis")
  return grouped_df8

def generateAggCountDf(df_copy, agg_columns):
  grouped_df9 = df_copy.groupby(['transcript_id',
                                 'transcript_position', 'Nucleotide Sequence']).size().reset_index(name='Read counts')

  return grouped_df9

"""#### Functions to feature engineer from nucleotide sequence"""


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
      # print('{} completed!'.format(pattern))
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

"""#### Function to combine all aggregations for dataframe"""

def generateAggDf(original_data):
  # Create a deep copy of the input DataFrame
  df_copy = original_data.copy()

  # Define the columns to aggregate
  agg_columns = df_copy.columns[3:12]

  # Mean
  grouped_df1 = generateAggMeanDf(df_copy, agg_columns)
  # Median
  grouped_df2 = generateAggMedianDf(df_copy, agg_columns)
  # Min
  grouped_df3 = generateAggMinDf(df_copy, agg_columns)
  # Max
  grouped_df4 = generateAggMaxDf(df_copy, agg_columns)
  # Std
  grouped_df5 = generateAggStdDf(df_copy, agg_columns)
  # Last
  grouped_df6 = generateAggLastDf(df_copy, agg_columns)
  # Skewness
  grouped_df7 = generateAggSkewnessDf(df_copy, agg_columns)
  # Kurtosis
  grouped_df8 = generateAggKurtosisDf(df_copy, agg_columns)
  # Read count
  grouped_df9 = generateAggCountDf(df_copy, agg_columns)

  # Merge the dataframes based on the specified criteria
  merged_df = pd.merge(grouped_df1, grouped_df2, on=['transcript_id', 'transcript_position', 'Nucleotide Sequence'])
  merged_df = pd.merge(merged_df, grouped_df3, on=['transcript_id', 'transcript_position', 'Nucleotide Sequence'])
  merged_df = pd.merge(merged_df, grouped_df4, on=['transcript_id', 'transcript_position', 'Nucleotide Sequence'])
  merged_df = pd.merge(merged_df, grouped_df5, on=['transcript_id', 'transcript_position', 'Nucleotide Sequence'])
  merged_df = pd.merge(merged_df, grouped_df6, on=['transcript_id', 'transcript_position', 'Nucleotide Sequence'])
  merged_df = pd.merge(merged_df, grouped_df7, on=['transcript_id', 'transcript_position', 'Nucleotide Sequence'])
  merged_df = pd.merge(merged_df, grouped_df8, on=['transcript_id', 'transcript_position', 'Nucleotide Sequence'])
  merged_df = pd.merge(merged_df, grouped_df9, on=['transcript_id', 'transcript_position', 'Nucleotide Sequence'])

  print("finish aggregation")

  # Nucleotide Seq cols
  final_df = createNucleotideSeqCols(merged_df)

  print("finish nucleotide seq")

  return final_df

def normalizeColumns(df):
    # Extract the columns to be normalized (columns 2 to 74)
    columns_to_normalize = df.columns[2:75]

    # Create a StandardScaler instance
    scaler = StandardScaler()

    # Normalize the selected columns
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

    return df

def write_to_csv(df, sample_name, df_type):
  file_name = '_'.join([sample_name, df_type, '.csv'])

  df.to_csv(os.path.join(os.getcwd(), 'final_data', file_name), index=False)

def run_aggregation():
  """
  Master function to aggregate files and write to csv
  """
  # get paths of all intermediate files
  file_names = {}
  preprocessed_folder_path = os.path.join(os.getcwd(), 'preprocessed_data')
  for f in os.listdir(preprocessed_folder_path):
      sample_name = '_'.join(f.split('_')[:-1])
      file_names[sample_name] = os.path.join(preprocessed_folder_path, f)

  cols_47 = ['transcript_id', 'transcript_position', '+1 sd signal_read_last', '-1 dwelling time_read_last',
               '+1 dwelling time_read_last', '-1 mean signal_read_skewness', '-1 mean signal_read_kurtosis',
               'central mean signal_read_kurtosis', '+1 mean signal_read_skewness', '-1 mean signal_read_std',
               'central mean signal_read_std', '+1 sd signal_read_min', 'central mean signal_read_skewness',
               '+1 mean signal_read_kurtosis', 'central dwelling time_read_last', '+1 dwelling time_read_min',
               '-1 dwelling time_read_min', 'Read counts', 'central sd signal_read_min', '+1 mean signal_read_std',
               'central dwelling time_read_min', 'AGACA_motif', 'AGACC_motif', 'AGACT_motif', 'AAACA_motif',
               'AAACC_motif', 'AAACT_motif', 'GGACA_motif', 'GGACC_motif', 'GGACT_motif', 'GAACA_motif',
               'GAACC_motif', 'GAACT_motif', 'TGACA_motif', 'TGACC_motif', 'TGACT_motif', 'TAACA_motif',
               'TAACC_motif', 'TAACT_motif', 'First_Nucleotide_A', 'First_Nucleotide_C', 'First_Nucleotide_G',
               'First_Nucleotide_T', 'Last_Nucleotide_A', 'Last_Nucleotide_C', 'Last_Nucleotide_G', 'Last_Nucleotide_T']

  # run functions for feature engineering and write to csv
  for sample_name, path in file_names.items():
    print(sample_name)
    original_df = pd.read_parquet(path)
    agg_df = generateAggDf(original_df)
    final_agg_df = normalizeColumns(agg_df) # should have 101 columns (excludes gene_id, label)
    final_47_df = final_agg_df[cols_47] # should have 47 columns (excludes gene_id, label)
    # write_to_csv(final_agg_df, sample_name, 'final_full')
    write_to_csv(final_47_df, sample_name, 'final_47')

if __name__ == "__main__":
    run_aggregation()

