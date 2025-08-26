### Simple Face Detection menggunakan OpenCV
*Disclaimer: Draft kode dan penjelasan ini dibuat oleh ChatGPT*

Berikut contoh skrip Python + OpenCV untuk membuka webcam dan mendeteksi banyak wajah secara realtime. Pakai Haar Cascade (ringan \& simpel). Saya sertakan juga opsi model DNN (lebih akurat) kalau mau upgrade.

#### 1\) Versi simpel (Haar Cascade)
```python
import cv2

def main():
    # 0 bisa diganti 1/2 jika punya beberapa kamera
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # di Linux/Mac bisa tanpa CAP_DSHOW

    if not cap.isOpened():
        print("Gagal membuka kamera. Coba ganti index ke 1 atau 2.")
        return

    # Load detektor wajah Haar Cascade (tersedia di OpenCV)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Opsi: atur resolusi (misal 1280x720). Boleh dikomentari kalau tak perlu.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Gagal membaca frame dari kamera.")
            break

        # Konversi ke grayscale agar deteksi lebih cepat
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Deteksi wajah (atur parameter sesuai kebutuhan)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,     # makin kecil -> lebih sensitif, tapi bisa lambat/false positive
            minNeighbors=5,      # makin besar -> lebih ketat (lebih sedikit false positive)
            minSize=(60, 60)     # ukuran minimum wajah
        )

        # Gambar kotak di setiap wajah yang terdeteksi
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Tampilkan jumlah wajah terdeteksi
        cv2.putText(frame, f"Faces: {len(faces)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Webcam - Face Detection (Haar)", frame)

        # Tekan 'q' untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```
Cara pakai:
* Install: pip install opencv-python
* Jalankan: python face\_detect\_haar.py
* Tekan q untuk keluar.

#### 2\) Versi lebih akurat (DNN ResNet SSD) — opsional
Unduh file model:
* Prototxt: deploy.prototxt (SSD face deploy)
* Caffe model: res10\_300x300\_ssd\_iter\_140000.caffemodel
Lalu:

```python
import cv2
import numpy as np

PROTOTXT = "deploy.prototxt"
MODEL = "res10_300x300_ssd_iter_140000.caffemodel"
CONF_THRESHOLD = 0.5

def main():
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Gagal membuka kamera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                     scalefactor=1.0,
                                     size=(300, 300),
                                     mean=(104.0, 177.0, 123.0))  # mean BGR
        net.setInput(blob)
        detections = net.forward()

        count = 0
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence >= CONF_THRESHOLD:
                count += 1
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{confidence:.2f}", (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.putText(frame, f"Faces: {count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Webcam - Face Detection (DNN)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```
Tips cepat:
* Jika wajah tidak terdeteksi: naikkan scaleFactor jadi 1.05 (Haar) atau turunkan CONF\_THRESHOLD jadi 0.4 (DNN)
* Jika lag: kecilkan resolusi webcam (mis. 640×480) atau pakai cap.set(cv2.CAP\_PROP\_FPS, 30)
* Kalau punya GPU (OpenCV build with CUDA), DNN bisa dipercepat; kalau tidak, tetap oke di CPU untuk 300×300
* Ubah CascadeClassifier menjadi lbpcascade_frontalface.xml untuk menggunakan LBP menggantikan HaarCascade

Note:
* File lbpcascade_frontalface.xml and haarcascade_frontalface_alt.xml was taken from https://github.com/informramiz/Face-Detection-OpenCV
* File haarcascade_frontalface_default.xml was taken from https://github.com/medsriha/real-time-face-recognition
