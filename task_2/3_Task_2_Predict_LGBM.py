import pandas as pd
import os
import pickle
import lightgbm as lgb

def write_to_csv(df, sample_name):
    file_name = '_'.join([sample_name, 'predict_lgbm.csv'])

    df.to_csv(os.path.join(os.getcwd(), 'lgbm_prediction', file_name), index=False)

def run_predict():
    # load saved model
    with open('LGBM_tuned.pkl' , 'rb') as f:
        model = pickle.load(f)
    
    final_data_path = os.path.join(os.getcwd(), 'final_data')
    for f in os.listdir(final_data_path):
        sample_name = '_'.join(f.split('_')[:-1])
        print(sample_name)

        file_path = os.path.join(final_data_path, f)
        df = pd.read_parquet(file_path)

        full_cols = df.columns
        first_two_cols = ['transcript_id', 'transcript_position'] # take first 2 cols for prediction df
        cols = [col for col in full_cols if col not in ['gene_id', 'transcript_id', 'Nucleotide Sequence', 'label']]

        X_test = df[cols]
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        pred_df = df[first_two_cols]
        pred_df["score"] = y_pred_proba
        write_to_csv(pred_df, sample_name)
        print(f"finish prediction for {sample_name}")

if __name__ == "__main__":
    run_predict()
