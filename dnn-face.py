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
