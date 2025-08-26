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
    cap.set(cv2.CAP_PROP_FPS, 30)

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
