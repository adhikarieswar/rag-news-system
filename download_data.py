# download_data.py
import kagglehub

print("Downloading dataset...")
path = kagglehub.dataset_download("rmisra/news-category-dataset")
print(f"✅ Dataset downloaded to: {path}")