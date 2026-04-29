import kagglehub

# Download latest version
path = kagglehub.dataset_download("sadmansakibmahi/plant-disease-expert")

print("Path to dataset files:", path)