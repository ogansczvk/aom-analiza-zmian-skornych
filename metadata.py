import pandas as pd
import os

zdjecia = './zdjecia'
files_in_folder = os.listdir(zdjecia)
df = pd.read_csv('HAM10000_metadata.csv', skiprows = 1, header=None)
count = 0

for index, row in df.iterrows():
    kolumna0 = row[1] #nr zdjecia
    kolumna2 = row[2] #choroba
    kolumna3 = row[3] #sposob badania
    if (kolumna2 == 'mel') and (kolumna3 == 'histo'):
        #print(kolumna0)
        for filename in files_in_folder:
            if kolumna0 in filename:
                count += 1
                print(count)
        