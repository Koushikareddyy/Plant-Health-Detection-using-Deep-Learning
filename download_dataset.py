import kagglehub

# Download the latest version of the dataset
path = kagglehub.dataset_download("emmarex/plantdisease")

print("✅ Dataset downloaded successfully!")
print("📁 Path to dataset files:", path)
