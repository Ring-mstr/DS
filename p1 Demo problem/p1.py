import pandas as pd
print(pd.__version__)
import pandas as pd
# Read the dataset
df = pd.read_csv("./water_potability.csv")
# Display the first few rows of the dataset
print(df.head())
