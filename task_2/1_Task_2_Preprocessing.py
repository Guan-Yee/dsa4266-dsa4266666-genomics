import os
import json
import pandas as pd
from sklearn.preprocessing import StandardScaler

"""### Preprocess data for each sample"""

def generate_dataframe(data_json_path):
  """
  Parameters:
  all_data (list): list of json objects (dict) from generate_all_data() function

  Returns:
  pd.Dataframe: data.json represented in a dataframe
  """
  flat_data = []
  f = open(data_json_path, 'r')

  # Iterate through the each json format in the list
  for line in f:
    try:
      line = json.loads(line) # change json string to dict
      for transcript, values in line.items():
        for pos, subdata in values.items():
          for sequence, values in subdata.items():
            for val in values:
              flat_data.append([transcript, pos, sequence, *val])
    except json.JSONDecodeError:
      print(f"Skipping invalid JSON: {line}")

  columns = ['transcript_id', 'transcript_position', 'Nucleotide Sequence',
            '-1 dwelling time', '-1 sd signal','-1 mean signal',
            'central dwelling time', 'central sd signal','central mean signal',
            '+1 dwelling time', '+1 sd signal','+1 mean signal']
  df = pd.DataFrame(flat_data, columns=columns)

  df['transcript_position'] = df['transcript_position'].astype('int64')

  return df

def write_to_parquet(df, sample_name):
  file_name = '_'.join([sample_name, 'intermediate.parquet'])

  df.to_parquet(os.path.join(os.getcwd(), 'preprocessed_data', file_name), index=False)

def preprocess_all_data():
  """
  Master function preprocess data for each sample
  """
  samples_path = {}
  data_dir = os.path.join(os.getcwd(), 'data')

  # get folder path for each sample
  for folder in os.listdir(data_dir):
    samples_path[folder] = os.path.join(data_dir, folder)

  for sample_name, path in samples_path.items():
    data_json_path = os.path.join(path, 'data.json')
    print(data_json_path)
    print(sample_name)
    final_df = generate_dataframe(data_json_path)
    write_to_parquet(final_df, sample_name)

if __name__ == "__main__":
  preprocess_all_data()


