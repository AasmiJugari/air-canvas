import cv2
import numpy as np
import mediapipe as mp 
import random

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Set up video capture
cap = cv2.VideoCapture(0)
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Create blank canvases
canvas_size = 390
target_canvas = np.ones((canvas_size, canvas_size, 3), np.uint8) * 255
user_canvas = np.ones((canvas_size, canvas_size, 3), np.uint8) * 255

# Initialize variables
drawing = False
last_point = None
brush_size = 10
score = 0
show_score = False
min_contour_area = 300  # Minimum contour area to be considered as a valid drawing

# Draw shapes on the canvas

def draw_shape(canvas, shape_type):
    canvas[:] = 255  # Clear the canvas
    color = (0, 0, 0)  # Outline color
    
    if shape_type == "triangle":
        pts = np.array([[canvas.shape[1] // 2, canvas.shape[0] // 2 - 60],
                        [canvas.shape[1] // 2 - 50, canvas.shape[0] // 2 + 50],
                        [canvas.shape[1] // 2 + 50, canvas.shape[0] // 2 + 50]], np.int32)
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2)
        
    elif shape_type == "rectangle":
        top_left = (canvas_size // 4, canvas_size // 4)
        bottom_right = (canvas_size * 3 // 4, canvas_size * 3 // 4)
        cv2.rectangle(canvas, top_left, bottom_right, color, thickness=2)

    elif shape_type == "circle":
        center = (canvas_size // 2, canvas_size // 2)
        radius = canvas_size // 4
        cv2.circle(canvas, center, radius, color, thickness=2)

    elif shape_type == "pentagon":
        pts = np.array([[canvas.shape[1] // 2, canvas.shape[0] // 2 - 60],
                        [canvas.shape[1] // 2 - 55, canvas.shape[0] // 2 - 20],
                        [canvas.shape[1] // 2 - 35, canvas.shape[0] // 2 + 50],
                        [canvas.shape[1] // 2 + 35, canvas.shape[0] // 2 + 50],
                        [canvas.shape[1] // 2 + 55, canvas.shape[0] // 2 - 20]], np.int32)
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2)

    elif shape_type == "hexagon":
        pts = np.array([[canvas.shape[1] // 2, canvas.shape[0] // 2 - 60],
                        [canvas.shape[1] // 2 - 50, canvas.shape[0] // 2 - 30],
                        [canvas.shape[1] // 2 - 50, canvas.shape[0] // 2 + 30],
                        [canvas.shape[1] // 2, canvas.shape[0] // 2 + 60],
                        [canvas.shape[1] // 2 + 50, canvas.shape[0] // 2 + 30],
                        [canvas.shape[1] // 2 + 50, canvas.shape[0] // 2 - 30]], np.int32)
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2)
    
   

def process_frame(frame):
    global drawing, last_point

    # Convert the frame from BGR to RGB for processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Check if any hand landmarks are detected
    if results.multi_hand_landmarks:
        # Process only the first hand detected
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = hand_landmarks.landmark
        
        # Calculate the x and y coordinates for the index finger tip
        x = int(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * canvas_size)
        y = int(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * canvas_size)

        # Mirror the x-coordinate for horizontal drawing mirroring
        mirrored_x = canvas_size - x

        if drawing:
            # If the user is drawing and there is a previous point, draw a line
            if last_point:
                cv2.line(user_canvas, (last_point[0], last_point[1]), (mirrored_x, y), (0, 0, 0), brush_size)
            # Update the last point with the current position
            last_point = (mirrored_x, y)
        else:
            # Reset the last point if not drawing
            last_point = None

        # Draw hand landmarks on the frame for better visualization
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    return frame


def calculate_score():
    global user_canvas, target_canvas
    
    # Convert canvases to grayscale
    target_gray = cv2.cvtColor(target_canvas, cv2.COLOR_BGR2GRAY)
    user_gray = cv2.cvtColor(user_canvas, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to create binary images
    _, target_thresh = cv2.threshold(target_gray, 240, 255, cv2.THRESH_BINARY_INV)
    _, user_thresh = cv2.threshold(user_gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Find contours in both images
    contours_target, _ = cv2.findContours(target_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_user, _ = cv2.findContours(user_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours_target and contours_user:
        # Filter out small contours considered as scribbles
        contours_user = [cnt for cnt in contours_user if cv2.contourArea(cnt) > min_contour_area]
        
        if contours_user:
            # Use the largest contour in the user drawing
            user_contour = max(contours_user, key=cv2.contourArea)
            target_contour = max(contours_target, key=cv2.contourArea)
            
            # Approximate the shapes for better comparison
            user_contour = cv2.approxPolyDP(user_contour, 0.02 * cv2.arcLength(user_contour, True), True)
            target_contour = cv2.approxPolyDP(target_contour, 0.02 * cv2.arcLength(target_contour, True), True)
            
            # Detect shape types for target and user contours
            target_shape_type = detect_shape_type(target_contour)
            user_shape_type = detect_shape_type(user_contour)
            
            # Check if the detected shapes match
            if target_shape_type == user_shape_type:
                # Adjust the shape matching score based on shape type
                shape_score = cv2.matchShapes(target_contour, user_contour, cv2.CONTOURS_MATCH_I1, 0.0) * 100
                
                # Custom adjustments for different shapes
                if target_shape_type == "triangle":
                    shape_score *= 0.2  # Easier to get a higher score for triangles
                elif target_shape_type == "rectangle":
                    shape_score *= 1.2  # Rectangles tolerate more variation
                elif target_shape_type == "hexagon":
                    shape_score *= 2.2  # Rectangles tolerate more variation
                elif target_shape_type == "pentagon":
                    shape_score *= 2.2  # Rectangles tolerate more variation        
                
                else:
                    # Default for other shapes
                    shape_score *= 3.0
                
                shape_score = max(0, 100 - shape_score)  # Convert to percentage
                
                # Proportional scoring: Consider aspect ratio and position
                target_rect = cv2.boundingRect(target_contour)
                user_rect = cv2.boundingRect(user_contour)
                
                aspect_ratio_score = 100 - abs((target_rect[2] / target_rect[3]) - (user_rect[2] / user_rect[3])) * 100
                aspect_ratio_score = max(0, aspect_ratio_score)
                
                position_score = 100 - abs((target_rect[0] - user_rect[0]) + (target_rect[1] - user_rect[1])) * 100 / (canvas_size * 2)
                position_score = max(0, position_score)
                
                # Combine scores with stricter weights and scale to range 10-80
                final_score = 0.4 * shape_score + 0.4 * aspect_ratio_score + 0.3 * position_score
                
                # Scale the final score to be between 10 and 80
                final_score = max(10,20,30,40,50, min(100, final_score))
                
                return final_score  # Return the adjusted score
            else:
                return 20  # Minimum score if the shape types do not match
    return 20  # Minimum score if no valid contours are found





def detect_shape_type(contour):
    # Simple shape type detection based on contour approximation
    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
    num_sides = len(approx)
    
    if num_sides == 3:
        return "triangle"
    elif num_sides == 4:
        aspect_ratio = cv2.contourArea(contour) / (cv2.boundingRect(contour)[2] * cv2.boundingRect(contour)[3])
        if 0.9 <= aspect_ratio <= 1.1:
            return "rectangle"  # Approximately square
        else:
            return "rectangle"  # More general rectangle case
    elif len(approx) > 10:
        return "traingle"
    return "unknown"

def clear_canvas():
    global user_canvas
    user_canvas = np.ones((canvas_size, canvas_size, 3), np.uint8) * 255

# Draw a random shape as the target
# Add the new shapes to the list of possible shapes
shapes = ["triangle", "rectangle", "circle", "pentagon", "hexagon"]

target_shape = random.choice(shapes)
draw_shape(target_canvas, target_shape)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = process_frame(frame)

    # Resize frame for smaller display
    frame_resized = cv2.resize(frame, (canvas_size, canvas_size))

    # Create instruction canvas
    instruction_canvas = np.zeros((canvas_size, canvas_size, 3), np.uint8)
    cv2.putText(instruction_canvas, "Instructions:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(instruction_canvas, "Press 'd' to Draw", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(instruction_canvas, "Press 's' to Stop", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(instruction_canvas, "Press 'c' to Clear and to change the shape", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(instruction_canvas, "Press 'e' to Evaluate", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(instruction_canvas, "Draw as per shape", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(instruction_canvas, f"Target Shape: {target_shape}", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    


    

    # Combine all elements into one screen
    top_row = np.hstack((target_canvas, user_canvas))
    bottom_row = np.hstack((frame_resized, instruction_canvas))

    # Add score text
    if show_score:
        score = calculate_score()
        cv2.putText(bottom_row, f"Score: {score:.2f}%", (10, canvas_size - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    full_screen = np.vstack((top_row, bottom_row))

    # Display the combined screen
    cv2.imshow('Drawing Game', full_screen)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('d'):
        drawing = True
    elif key == ord('s'):
        drawing = False
    elif key == ord('e'):
        show_score = True
    elif key == ord('c'):
        clear_canvas()
        target_shape = random.choice(shapes)
        draw_shape(target_canvas, target_shape)
        show_score = False
        print("Canvas cleared.")

cap.release()
cv2.destroyAllWindows()
