import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)
hands = mp_hands.Hands(max_num_hands=1)

cap = cv2.VideoCapture(0)

# 👉 Set camera resolution (you can change)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 200)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 100)

mask_scale = 1.0

# 👉 Fullscreen window
cv2.namedWindow("AI Mask", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("AI Mask", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # -------- FACE --------
    face_results = face_mesh.process(rgb)

    face_landmarks_list = None
    face_center = None

    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            face_landmarks_list = face_landmarks

            nose = face_landmarks.landmark[1]
            cx, cy = int(nose.x * w), int(nose.y * h)
            face_center = (cx, cy)

    # -------- HAND --------
    hand_results = hands.process(rgb)

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:

            # 👉 Draw hand skeleton
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(255, 255, 0), thickness=1, circle_radius=2),
                mp_draw.DrawingSpec(color=(255, 0, 0), thickness=1)
            )

            # Thumb (4) & Index (8)
            thumb = hand_landmarks.landmark[4]
            index = hand_landmarks.landmark[8]

            x1, y1 = int(thumb.x * w), int(thumb.y * h)
            x2, y2 = int(index.x * w), int(index.y * h)

            # Draw points
            cv2.circle(frame, (x1, y1), 6, (0, 255, 0), -1)
            cv2.circle(frame, (x2, y2), 6, (0, 255, 0), -1)

            # Distance
            dist = np.hypot(x2 - x1, y2 - y1)

            # Control mask size
            mask_scale = dist / 100

    # -------- SIDE FACE MESH --------
    if face_landmarks_list and face_center:
        cx, cy = face_center

        offset_x = int(200 * mask_scale)
        side_cx = cx + offset_x
        side_cy = cy

        for lm in face_landmarks_list.landmark:
            x = int(lm.x * w)
            y = int(lm.y * h)

            new_x = int(side_cx + (x - cx) * mask_scale)
            new_y = int(side_cy + (y - cy) * mask_scale)

            cv2.circle(frame, (new_x, new_y), 1, (0, 0, 0), -1)

    cv2.imshow("AI Mask", frame)

    key = cv2.waitKey(1)

    # ESC to exit
    if key == 27:
        break

    # Press 'f' to toggle fullscreen
    if key == ord('f'):
        cv2.setWindowProperty("AI Mask", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Press 'w' for window mode
    if key == ord('w'):
        cv2.setWindowProperty("AI Mask", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

cap.release()
cv2.destroyAllWindows()