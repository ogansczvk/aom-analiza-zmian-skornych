import os
import shutil
import pandas as pd
import random

# Ustawienie losowości (dla powtarzalnych wyników)
random.seed(42)

# Ścieżki
metadata_path = r'C:\Users\oliwi\OneDrive\Pulpit\Studia\semestr 6\Analiza Obrazów Medycznych\projekt\data\HAM10000_metadata.csv'
data_dir1 = r'C:\Users\oliwi\OneDrive\Pulpit\Studia\semestr 6\Analiza Obrazów Medycznych\projekt\data\HAM10000_images_part_1'
data_dir2 = r'C:\Users\oliwi\OneDrive\Pulpit\Studia\semestr 6\Analiza Obrazów Medycznych\projekt\data\HAM10000_images_part_2'

# Wczytaj metadane
df = pd.read_csv(metadata_path)

# Filtruj klasy
mel_df = df[df['dx'] == 'mel']
nonmel_df = df[df['dx'] != 'mel']

# Zbalansuj liczebność (np. 1:1)
nonmel_df = nonmel_df.sample(n=len(mel_df), random_state=42)

#Połącz i pomieszaj
binary_df = pd.concat([mel_df, nonmel_df]).sample(frac=1, random_state=42).reset_index(drop=True)

# Podział na 80% trening / 20% test
split_idx = int(0.8 * len(binary_df))
train_df = binary_df[:split_idx]
test_df = binary_df[split_idx:]

#Funkcja do kopiowania do struktury
def copy_images(df, target_root):
    for _, row in df.iterrows():
        label = 'melanoma' if row['dx'] == 'mel' else 'not_melanoma'
        filename = row['image_id'] + '.jpg'

        src = os.path.join(data_dir1, filename) if os.path.exists(os.path.join(data_dir1, filename)) \
              else os.path.join(data_dir2, filename)

        dst_dir = os.path.join(target_root, label)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy(src, os.path.join(dst_dir, filename))

# Czyszczenie docelowych folderów
for base in ['czerniaki_binary_train', 'czerniaki_binary_test']:
    if os.path.exists(base):
        shutil.rmtree(base)

# Kopiowanie zdjęć
copy_images(train_df, 'czerniaki_binary_train')
copy_images(test_df, 'czerniaki_binary_test')

print("Gotowe: utworzono strukturę binary classification (melanoma vs not_melanoma)")
