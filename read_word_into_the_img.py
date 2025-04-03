import easyocr as ocr
import cv2
import cv2.data
from matplotlib import pyplot as plt
import numpy as np

ocr_motoru = ocr.Reader(['en', 'tr']) # ocr.Reader -> okuyucuyu sunuyor

with open("nusret.jpg", "rb") as f:
    img = f.read()

img_ham = np.fromstring(img, np.uint8)
img_cv2 = cv2.imdecode(img_ham, cv2.IMREAD_COLOR)

words = ocr_motoru.readtext(img)

with open("words.txt", "w") as file:
   for word in words:
      file.write(f"{word[1]}\n")


for word in words:
  cv2.rectangle(img_cv2, 
                [int(word[0][0][0]), int(word[0][0][1])], 
                [int(word[0][2][0]), int(word[0][2][1])], 
                (0, 0, 255), 2) #rectangle -> dikdörtgeni çiz. word -> dikdörtgenin iki köşesini alıyoruz. 0,0,255 renk olarak al, kırmızı 255. 2 -> kalınlık

img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)

plt.imshow(img_rgb)
plt.axis('off')  
plt.show()
cv2.imwrite("nusret_ocr_sonuc.jpg", img_cv2)

# -----------------------------------------------------------------------------------------------------------------------------
# görüntü olarak elde edip göstermek istemezsek, yalnızca görüntü içindeki kelimeleri seçerek işlem yapmak istersek:
# ocr_motoru = ocr.Reader(['en', 'tr'])
# words = ocr_motoru.readtext(img)
# print(words)
# -----------------------------------------------------------------------------------------------------------------------------

 