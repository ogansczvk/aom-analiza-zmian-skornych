import cv2
import os
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import pickle


# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])

# #IMAGE ACQUISITION

# count = 0
# max_count = 10

# folder_path = os.path.join(os.path.dirname(__file__), 'zdjecia')
# for filename in os.listdir(folder_path):
#     if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
#         file_path = os.path.join(folder_path, filename)
#         image = Image.open(file_path)
#         image_np = np.array(image)
#         img=image_np[30:410,30:560]
#         grayScale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#         kernel = cv2.getStructuringElement(1,(15,15)) # maska o wymiarach 9x9 #1 oznacza krzyżowy element strukturalny, 0 -prostokatny, 2-eliptyczny
#         blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
#         tophat = cv2.morphologyEx(blackhat, cv2.MORPH_TOPHAT, kernel)
#         bhg= cv2.GaussianBlur(tophat,(3,3),cv2.BORDER_DEFAULT)
#         ret,mask = cv2.threshold(bhg,10,255,cv2.THRESH_BINARY)
#         dst = cv2.inpaint(img,mask,6,cv2.INPAINT_TELEA)
#         image_tensor = transform(image).unsqueeze(0)

#         print(f"[{count+1}] Przetworzono: {filename} → tensor shape: {image_tensor.shape}")
#         cv2.imshow("dst",dst)
#         count += 1
#         if count >= max_count:
#             print(f"\nZatrzymano po {max_count} obrazach (test OK ✅)")
#             break
        

count = 0

with open("photos.pkl", "rb") as f:
    photos = pickle.load(f)

#tworzenie folderu na nowe zdjecia
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

#Input image
for path in photos:

#Read image
    image=cv2.imread(path,cv2.IMREAD_COLOR)
#Image cropping
    img=image[30:410,30:560]

#DULL RAZOR (REMOVE HAIR) - metoda

#Gray scale
    grayScale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY )
#Black hat filter - wykrywanie ciemnych obiektów
    kernel = cv2.getStructuringElement(1,(15,15)) # maska o wymiarach 9x9 #1 oznacza krzyżowy element strukturalny, 0 -prostokatny, 2-eliptyczny
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    tophat = cv2.morphologyEx(blackhat, cv2.MORPH_TOPHAT, kernel)

#Gaussian filter
    bhg= cv2.GaussianBlur(tophat,(3,3),cv2.BORDER_DEFAULT)
#tu musza byc wartosci nieparzyste, przy zwiekszeniu do (9,9) daje gorsze efekty
#Binary thresholding (MASK) Jeśli piksel ma wartość większą niż 10, staje się białym (255), w przeciwnym razie czarnym (0).
    ret,mask = cv2.threshold(bhg,10,255,cv2.THRESH_BINARY)
#Replace pixels of the mask - Metoda "inpainting" zastępuje obszary z włosami pobliskimi pikselami skóry.
    dst = cv2.inpaint(img,mask,6,cv2.INPAINT_TELEA)
    count += 1
    output_path = os.path.join(output_dir, f"processed_{count}.jpg")
    cv2.imwrite(output_path, dst)

print(f"Przetworzono {count} obrazów z {len(photos)}.")

#Display images
#    cv2.imshow("Original image",image)
#    cv2.imshow("Cropped image",img)
#    cv2.imshow("Gray Scale image",grayScale)
#   cv2.imshow("tophat",tophat)
#    cv2.imshow("blackhat", blackhat)
#    cv2.imshow("Binary mask",mask)
#   cv2.imshow("Clean image tophat",dst)
#cv2.imshow("Clean image",dst_blackhat)

#    cv2.waitKey()
#cv2.destroyAllWindows()
