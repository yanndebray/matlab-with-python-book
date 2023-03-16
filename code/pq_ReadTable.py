import pandas as pd
import os 

# change to current directory
thisDirectory = os.path.dirname(os.path.realpath(__file__))
os.chdir(thisDirectory)

# read parquet file via pyarrow library
df = pd.read_parquet('newdata.parquet')
print(df)
