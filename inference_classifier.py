import pickle
import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'good', 1: 'hello', 2: 'home', 3: 'meet', 4: 'thanks'}

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        # Ensure the input data is a 2D array with 84 features
        if len(data_aux) == 42:
            data_aux += [0] * 42  # Pad with zeros to make it 84 features

        prediction = model.predict([data_aux])

        # Debugging print statements
        print("Raw prediction:", prediction)

        # Ensure the prediction is an integer
        try:
            predicted_label_index = int(prediction[0])
        except ValueError:
            print("Predicted label:", prediction[0])
        
        # Modify the rectangle coordinates based on your application logic
        x = int(min(x_) * W) - 10
        y = int(min(y_) * H) - 10
        x_ = int(max(x_) * W) - 10
        y_ = int(max(y_) * H) - 10

        cv2.rectangle(frame, (x, y), (x_, y_), (0, 0, 0), 4)
        cv2.putText(frame, prediction[0], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()