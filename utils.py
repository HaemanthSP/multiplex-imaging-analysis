import os
import re
import sys
import pandas as pd


def combine_feature_files(input_dir, feature_files, region_list=None):   
    if region_list is not None:
        if len(region_list) != len(feature_files):
            sys.exit("Total number of feature files and region alias should match")

    def invalid_column(column_name):
        pattern = r"DAPI C\d+ Biomarker Exp"
        return re.search(pattern, column_name) is None

    df = pd.DataFrame()
    # concat old dataframes
    for i in range(len(feature_files)):
        print(i)
        tmp = pd.read_csv(os.path.join(input_dir, feature_files[i]))
        tmp["region_num"] = str(i)
        if region_list is not None:
            tmp["unique_region"] = str(region_list[i])
            # tmp["Cell Id"] = tmp["Cell Id"] + "-" + tmp["unique_region"]
        df = pd.concat([df, tmp], ignore_index=True, axis=0)

    invalid_columns = [x for x in df.columns if not invalid_column(x)]
    valid_columns = df.columns.difference(invalid_columns)

    print(f"Invalid columns: {invalid_columns}")

    df = df[valid_columns].copy()
    print(f"Colums mismatch: {df.columns[df.isna().any()].tolist()}")

    # if df[df.isna().any(axis=1)] is not []:
    #     print(f"Colums mismatch: {df.columns[df.isna().any()].tolist()}")
    #     raise Exception("File contains NaN")
    return df

def save_cluster_files():
    pass