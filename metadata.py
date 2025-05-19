import os
import zipfile
import pandas as pd
import os
import pickle
from kaggle.api.kaggle_api_extended import KaggleApi

download_path = "./data"



if not os.path.exists(download_path):
    os.makedirs(download_path)

#sciezka do pliku kaggle.json
os.environ['KAGGLE_CONFIG_DIR'] = os.path.expanduser('C:\Users\oliwi\.kaggle\kaggle.json') 

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

zdjecia = './zdjecia'
files_in_folder = os.listdir(zdjecia)
df = pd.read_csv('HAM10000_metadata.csv', skiprows = 1, header=None)
count = 0

for index, row in df.iterrows():
    kolumna0 = row[1] #nr zdjecia kod isic
    kolumna2 = row[2] #choroba
    kolumna3 = row[3] #sposob badania
    if (kolumna2 == 'mel') and (kolumna3 == 'histo'): #mel ma tylko histo
        #print(kolumna0)
        for filename in files_in_folder:
            if kolumna0 in filename:
                photo_path = os.path.join(zdjecia, filename)
                photos.append(photo_path)
                count += 1
                photo = kolumna0
                print(count)

# Zapis listy ścieżek do pliku
with open("photos.pkl", "wb") as f:
    pickle.dump(photos, f)
        