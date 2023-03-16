import pandas as pd
import numpy as np
import os 

# change to current directory
thisDirectory = os.path.dirname(os.path.realpath(__file__))
os.chdir(thisDirectory)

# create dataframe
df = pd.DataFrame({'column1': [-1, np.nan, 2.5], 
'column2': ['foo', 'bar', 'tree'], 
'column3': [True, False, True]})
print(df)

# save dataframe to parquet file via pyarrow library
df.to_parquet('data.parquet', index=False)
