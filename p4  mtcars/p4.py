import pandas as pd
import matplotlib.pyplot as plt
# Load the dataset
mtcars = pd.read_csv('./mtcars.csv')
# Plot histogram
plt.hist(mtcars['mpg'], bins=10, color='skyblue', edgecolor='black')
# Add labels and title
plt.xlabel('Miles per gallon')
plt.ylabel('Frequency')
plt.title('Frequency Distribution of Miles per Gallon')
# Show plot
plt.show()
