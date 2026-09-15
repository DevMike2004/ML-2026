import pandas as pd 
import matplotlib as mt
import kagglehub

# Download latest version
download = kagglehub.dataset_download("dnkumars/cybersecurity-intrusion-detection-dataset")

print("Path to dataset files:", download)

path = f"{download}/cybersecurity_intrusion_data.csv"

df = pd.read_csv(path)

print("---------------------------------------\n")
df.describe()
print("---------------------------------------\n")
df.info()
print("---------------------------------------\n")
df.head()
