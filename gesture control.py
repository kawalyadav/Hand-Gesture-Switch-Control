import cv2
import numpy as np

cap = cv2.VideoCapture(0)
    
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    x1, y1 = 150, 150
    x2, y2 = 350, 350
    roi = frame[y1:y2, x1:x2]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)

    lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)

    mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.GaussianBlur(mask, (5,5), 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    text = "No Hand Detected"

    if len(contours) > 0:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)

        if area > 4000:

            hull = cv2.convexHull(cnt)
            hull_indices = cv2.convexHull(cnt, returnPoints=False)

            finger_count = 0

            if hull_indices is not None and len(hull_indices) > 3:
                defects = cv2.convexityDefects(cnt, hull_indices)

                if defects is not None:
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i][0]

                        start = tuple(cnt[s][0])
                        end = tuple(cnt[e][0])
                        far = tuple(cnt[f][0])

                        a = np.linalg.norm(np.array(end) - np.array(start))
                        b = np.linalg.norm(np.array(far) - np.array(start))
                        c = np.linalg.norm(np.array(end) - np.array(far))

                        if b * c == 0:
                            continue

                        angle = np.degrees(np.arccos((b**2 + c**2 - a**2) / (2*b*c)))

                        if angle < 80 and d > 8000:
                            finger_count += 1

            if finger_count == 0:
                x, y, w, h = cv2.boundingRect(cnt)

                if h > w * 1.2:
                    text = "Light ON (1 Finger)"
                else:
                    text = "Light OFF (Fist)"
            else:
                text = f"{finger_count + 1} Fingers"

        else:
            text = "Place Hand Properly"

    cv2.putText(frame, text, (40, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0,255,0), 2)

    cv2.imshow("Mask", mask)
    cv2.imshow("Improved Gesture Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
