import cv2
import os
import time
import sys

# Create a directory to store the images
if not os.path.exists("data"):
    os.makedirs("data")

# Define the expressions and their corresponding directories
expressions = {
    "eyebrow_raised": "pawn_forward",
    "eyebrow_frown": "pawn_diagonal_left",
    "eyebrow_smile": "pawn_diagonal_right",
    "head_tilt_up": "pawn_promotion",
    "smile": "knight_move",
    "frown": "bishop_move",
    "head_tilt_left": "rook_horizontal_left",
    "head_tilt_right": "rook_horizontal_right",
    "pucker_lips": "queen_move",
    "head_nod": "king_move",
    "head_nod_left": "castling_left",
    "head_nod_right": "castling_right"
}

# Create directories if they do not exist
for expression in expressions.values():
    if not os.path.exists(expression):
        os.makedirs(expression)

# Start video capture
cap = cv2.VideoCapture(0)

for expression, folder in expressions.items():
    print(f"Please perform the expression for: {expression}")
    time.sleep(3)  # Give the user time to prepare

    count = 0
    while count < 50:  # Capture 50 images per expression
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Video', gray)
        
        # Save the frame
        img_name = os.path.join(folder, f"{expression}_{count}.png")
        cv2.imwrite(img_name, gray)
        
        count += 1
        
        # Display the frame
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

print("Data collection complete.")
cap.release()
cv2.destroyAllWindows()
