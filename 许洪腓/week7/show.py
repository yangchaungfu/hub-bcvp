import pandas as pd 
from config import Config
df = pd.read_csv(Config['result_path'])
print(df)