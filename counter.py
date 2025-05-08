import os

data1 = r'C:\Users\olasu\projekty\aom-analiza-zmian-skornych\data\HAM10000_images_part_1'
data2 = r'C:\Users\olasu\projekty\aom-analiza-zmian-skornych\data\HAM10000_images_part_2'

# Liczymy pliki obrazów (np. JPG)
sum1 = len([f for f in os.listdir(data1) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
sum2 = len([f for f in os.listdir(data2) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

total = sum1 + sum2

print(f"Zdjęć w part 1: {sum1}")
print(f"Zdjęć w part 2: {sum2}")
print(f"Razem zdjęć: {total}")
