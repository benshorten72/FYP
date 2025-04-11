import pandas as pd
import sys

filenames = ["fullintefer/timestamp.txt","interferencesemi/timestamp.txt","nointeference/timestamp.txt"]
for filename in filenames:
    df = pd.read_csv(filename, header=None, names=["col1", "col2"], parse_dates=[0, 1])

    df['diff'] = (df['col1'] - df['col2']).abs()

    average_diff_seconds = df['diff'].mean().total_seconds()

    print(f"Average time difference: {average_diff_seconds:.3f} seconds for",filename)

