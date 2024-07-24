import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('expression_model.h5')

# Load label map
label_map = {
    0: "pawn_forward",
    1: "pawn_diagonal_left",
    2: "pawn_diagonal_right",
    3: "pawn_promotion",
    4: "knight_move",
    5: "bishop_move",
    6: "rook_horizontal_left",
    7: "rook_horizontal_right",
    8: "queen_move",
    9: "king_move",
    10: "castling_left",
    11: "castling_right"
}

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face / 255.0
        face = np.reshape(face, (1, 48, 48, 1))

        # Predict expression
        prediction = model.predict(face)
        move_idx = np.argmax(prediction)
        move = label_map[move_idx]

        # Draw bounding box and move
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, move, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
