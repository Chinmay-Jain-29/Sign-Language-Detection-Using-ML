import cv2
import mediapipe as mp
import numpy as np
import pickle

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
label_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'K',10:'L',11:'M',12:'N',13:'O',14:'P',15:'Q',16: 'R', 17: 'S', 18: 'T',19:'U',20:'V',21:'W',22:'X',23:'Y'}

while True:
    data_aux = []

    ret, frame = cap.read()
    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.putText(frame, 'SIGN LANGUAGE DETECTION', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 250), 3,
                cv2.LINE_AA)
    cv2.putText(frame, 'CHINMAY PRAKASH JAIN', (12, 83), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (225, 0, 0), 1,
                cv2.LINE_AA)
    cv2.putText(frame, 'SAMIKSHA NAMDEV MORE', (12, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (225, 0, 0), 1,
                cv2.LINE_AA)
    cv2.putText(frame, 'PURVA KISHOR AHIRE', (12, 121), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (225, 0, 0), 1,
                cv2.LINE_AA)
    cv2.putText(frame, 'RAHUL HARISHCHANDRA SOMVANSHI', (12, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (225, 0, 0), 1,
                cv2.LINE_AA)


    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                data_aux.append(x)
                data_aux.append(y)

        x1 = int(min(data_aux[::2]) * W)
        y1 = int(min(data_aux[1::2]) * H)

        x2 = int(max(data_aux[::2]) * W)
        y2 = int(max(data_aux[1::2]) * H)

        # Convert data_aux to a NumPy array and reshape it
        data_aux_array = np.asarray(data_aux).reshape(1, -1)

        # Make the prediction
        prediction = model.predict(data_aux_array)

        predicted_character = label_dict[int(prediction[0])]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        print(predicted_character)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()