import cv2
import cv2.data
import numpy as np 

# Video için dahil olan kamerayı kullan (0) , usb ile kamera kullanıyorsan (1) , herhangi bir videoyu açmak için (2) yazılacak
cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

face_count = 0

while True:
    # kamerayı okuması için read komutu eklenir. ret ve frame olarak iki değer veriliyor,
    # frame ilk görüntüyü alıp frame değişkenine atar
    ret, frame = cap.read()

    # kameradaki görüntüyü grileştirir. Çünkü bilgisayar daha rahat okur
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # yüzleri tanımlamak için yazılan kod
    faces = face_cascade.detectMultiScale(grey, scaleFactor=1.3, minNeighbors=5) # scaleFactor = yüz tespit işlemi sırasında ölçeklendirme oranını belirliyor. Daha küçük değer (1.1) daha fazla tespit demektir. Ancak bu durum daha fazla işşlem gücü gerektirir ve daha yavaş çalışabilir.
    # minNeighbors = 'detectMultiScale' fonksiyonunda algılanan bir bölgenin yüz olarak kabul edilmesi için gerekli olan dikdörtgenlerin sayısını belirtiyor
    
    face_count += len(faces)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (85, 255, 0), 3) # cv2.rectangle = görüntü üzerine dikdörtgen çizmek için. tespit edilen yüzlerin veya diğer nesnelerin etrafını vurgulamak için kullanılır
    # frame = orjinal görüntü
    # (x,y) = dikdörtgenin sol üst köşe koordinatları
    # (x+w, y+h) = dikdörtgenin sağ alt koordinatları
    # (85, 255, 0) = RGB renk formatında dikdörtgenin rengi
    # 3 = çizgi kalınlığı
    print(faces)
    
    # Renklendirilen frame'i göster
    cv2.imshow("video", frame)
    
    # eğer "q" ya basarsak kamerayı kapatır yani programı durur. 0xFF==ord("q") bu kod kapatmak için yazılan bir kod.
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

print(face_count)
cap.release()
cv2.destroyAllWindows()