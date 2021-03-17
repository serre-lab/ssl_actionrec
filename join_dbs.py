
import os
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--db_path", type=str, default="../ssl_db")
    parser.add_argument("--db_save", type=str, default="dbs_pt.csv")

    args = parser.parse_args()
    dbs = os.listdir(args.db_path)

    df = pd.DataFrame()

    for db in dbs:
        df = df.append(pd.read_csv(os.path.join(args.db_path, db), index_col=0), ignore_index=True)
    
    df.to_csv(args.db_save)
    

if __name__ == "__main__":
    main()