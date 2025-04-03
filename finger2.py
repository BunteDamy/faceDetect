import cv2
import cv2.data
import numpy as np 


# Video için dahil olan kamerayı kullan (0), usb ile kamera kullanıyorsan (1), herhangi bir videoyu açmak için (2) yazılacak
cap = cv2.VideoCapture(0)  # Kamerayı başlat

if not cap.isOpened():
    print("Kamera açılamadı.")
    exit()
while True:
    ret, frame = cap.read()
    if not ret:
        print("Kare alınamıyor.")
        break

    cv2.imshow("Kamera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# Yüz tanıma modeli. CascadeClassifier ==> yüz, göz, insan gibi nesneleri tespit etmek için kullanılır
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

# el tanıma fonksiyonu
def detect_fingers(contour):
    hull = cv2.convexHull(contour, returnPoints=False) # dışbükey kapanın indeksi
    defects = cv2.convexityDefects(contour, hull) # convexityDefects = içbükey kusurlar için kullanılıyor. parmak çukurlarına denk geliyor
    if defects is None: 
        return 0
    finger_count = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(contour[s][0]) # parmak başlangıç
        end = tuple(contour[e][0]) # parmak bitiş
        far = tuple(contour[f][0]) # derin nokta
        a = np.linalg.norm(np.array(start) - np.array(end)) # bu üç nokta
        b = np.linalg.norm(np.array(start) - np.array(far)) # arasında oluşan
        c = np.linalg.norm(np.array(end) - np.array(far))   # üçgenin kenar uzunlukları
        angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # Cosine Teoremi = üçgenin iç açısı hesaplanıyor

        if angle <= np.pi / 2:  # İki parmağın arasındaki açı 90 derece veya daha azsa, parmak olarak kabul et
            finger_count += 1  # hesaplanan parmağı sayma sayısına ekliyorum

    return finger_count + 1 # ÇALIŞTIRIRKEN BURDAKİ +1 SİL ÖYLE DENE Bİ

            
# Yüz tespit sayacını başlat
face_count = 0
stop_signal_detected = False
consecutive_two_finger_frames = 0  # Arka arkaya iki parmak tespit edilen kare sayısı

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
        cv2.rectangle(frame, (x, y), (x+w, y+h), (85, 255, 0), 3)
    # frame = orjinal görüntü
    # (x,y) = dikdörtgenin sol üst köşe koordinatları
    # (x+w, y+h) = dikdörtgenin sağ alt koordinatları
    # (85, 255, 0) = RGB renk formatında dikdörtgenin rengi
    # 3 = çizgi kalınlığı
    
    # ----------- EL TESPİTİ VE PARMAK SAYISI --------------
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # HSV renk tespitinde daha güvenilir. aydınlatma olaylarına daha dayanıklı. BGR'den HSV'ye dönüştürüyoruz görüntüyü.
    lower_skin = np.array([0, 20, 70], dtype=np.uint8) # cilt tonu
    upper_skin = np.array([20, 255, 255], dtype=np.uint8) # cilt tonu
    mask = cv2.inRange(hsv, lower_skin, upper_skin) # aralıktaki piksel değerlerini maskeleyerek cilt rengi içeren alanları beyaz, geri kalan kısmı siyah yapar
    
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=4) # cilt rengini daha belirgin hale getirmek için maskeyi genişletir (dilate)
    mask = cv2.GaussianBlur(mask, (5, 5), 100) # maske üzerinde pürüzsüzleştirme yapmak için Gauss bulanıklaştırma yapılıyor
    
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # findContours = maskelenmiş görüntüdeki beyaz alanların konturlarını bulmak için. Böylece cilt rengine uygun olarak maskelenmiş elin sıırlarını belirliyor
    
    if contours:
        max_contour = max(contours, key=cv2.contourArea) # en büyük alanı kaplayan kontur -> genellikle bu el olur
        hull = cv2.convexHull(max_contour) # convexHull = dışbükey örtüsü. parmak uçları bu dışbükey kapanının kenarlarına denk geliyor
        cv2.drawContours(frame, [max_contour], -1, (0, 255, 255), 2)
        cv2.drawContours(frame, [hull], -1, (0, 255, 0), 2)
        
        finger_count = detect_fingers(max_contour) # parmak sayısı için fonksiyona git
        
        # Parmak sayısını ekrana yaz
        cv2.putText(frame, f'Fingers: {finger_count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        # Arka arkaya iki parmak tespit edilen kare sayısını artır
        if finger_count == 3:
            consecutive_two_finger_frames += 1
        else:
            consecutive_two_finger_frames = 0
        
        # Eğer iki parmak tespiti yeterince uzun süre devam ederse durdurma sinyalini etkinleştir
        if consecutive_two_finger_frames > 20:  # 20 kare boyunca iki parmak tespit edilirse
            stop_signal_detected = True

    cv2.imshow("video", frame)

    if stop_signal_detected or cv2.waitKey(1) & 0xFF == ord("q"): # q tuşuna basılınca kamera kapatılıyor.
        break

cap.release()
cv2.destroyAllWindows()

print(f"Tespit edilen toplam yüz sayısı: {face_count}")
