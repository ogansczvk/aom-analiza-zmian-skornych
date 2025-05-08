import os
import zipfile
import pandas as pd
import pickle
from kaggle.api.kaggle_api_extended import KaggleApi

download_path = "./data"



if not os.path.exists(download_path):
    os.makedirs(download_path)

#sciezka do pliku kaggle.json
os.environ['KAGGLE_CONFIG_DIR'] = os.path.expanduser('C:/Users/olasu/.kaggle') 

#pobranie danych z kaggle
dataset = "kmader/skin-cancer-mnist-ham10000"
download_path = "./data"

if not os.path.exists(download_path):
    os.makedirs(download_path)

api = KaggleApi()
api.authenticate()

# Pobieranie ZIP
print("Pobieranie zbioru danych z Kaggle...")
api.dataset_download_files(dataset, path=download_path, unzip=False)

# Wypakowywanie ZIP
zip_path = os.path.join(download_path, "skin-cancer-mnist-ham10000.zip")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(download_path)
print("✅ Dane wypakowane do:", download_path)

# === Wczytywanie metadanych i wybieranie czerniaka (mel) z histo ===
metadata_path = os.path.join(download_path, "HAM10000_metadata.csv")
images_folder = download_path
photos = []

df = pd.read_csv(metadata_path)
count = 0

print("Wyszukiwanie zdjęć czerniaka z histopatologią...")

for _, row in df.iterrows():
    img_id = row['image_id']
    dx = row['dx']
    method = row['dx_type']
    if dx == 'mel' and method == 'histo':
        filename = img_id + ".jpg"
# Szukamy pliku rekursywnie w ./data
        photo_path = None
        for root, dirs, files in os.walk(images_folder):
            if filename in files:
                photo_path = os.path.join(root, filename)
                break

        if photo_path:
            photos.append(photo_path)

            count += 1
            print(f"[{count}] Dodano: {filename}")

# Zapis listy ścieżek do pliku
with open("photos.pkl", "wb") as f:
    pickle.dump(photos, f)

print(f"📦 Zapisano {len(photos)} ścieżek do pliku photos.pkl")
