import cv2
import mediapipe as mp
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Initialize video capture
cap = cv2.VideoCapture(0)

# Flags and counters
photo_counter = 1
hand_detected = False
capture_delay = 2  # Delay time in seconds

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if not hand_detected:
                hand_detected_time = time.time()
                hand_detected = True

            break  # Exit the loop if a hand is detected
    else:
        hand_detected = False

    if hand_detected:
        elapsed_time = time.time() - hand_detected_time

        if elapsed_time < capture_delay:
            time_remaining = int(capture_delay - elapsed_time)
            cv2.putText(frame, f'Capturing photo in {time_remaining}...', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            # Capture and save the image
            photo_filename = f'selfie_{photo_counter}.png'
            cv2.imwrite(photo_filename, frame)
            print(f'Selfie saved as {photo_filename}')
            photo_counter += 1  # Increment photo counter for next image
            hand_detected = False  # Reset hand detected flag
            cv2.putText(frame, 'Photo Taken!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    else:
        cv2.putText(frame, 'Show Hand', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Selfie Camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
