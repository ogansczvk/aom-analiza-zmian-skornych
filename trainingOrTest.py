import pickle
import random
import os
import shutil



# Wczytaj listę zdjęć
with open("photos.pkl", "rb") as f:
    photos = pickle.load(f)

# Pomieszaj losowo dane
random.shuffle(photos)

# Podział 80/20
split_index = int(0.8 * len(photos))
train_photos = photos[:split_index]
test_photos = photos[split_index:]

print(f" Trening: {len(train_photos)} | Test: {len(test_photos)}")

# Foldery docelowe
train_dir = "./czerniaki_train"
test_dir = "./czerniaki_test"

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Wyczyść istniejące foldery
for folder in [train_dir, test_dir]:
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)


# Kopiowanie zdjęć do odpowiednich folderów
for path in train_photos:
    shutil.copy(path, os.path.join(train_dir, os.path.basename(path)))

for path in test_photos:
    shutil.copy(path, os.path.join(test_dir, os.path.basename(path)))

print("Podział i kopiowanie zakończone.")
