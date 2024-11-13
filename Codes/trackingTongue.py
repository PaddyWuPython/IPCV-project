import cv2 as cv
import dlib
import numpy as np

# Load dlib's face detector and 68 facial landmark predictor model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Ensure this model is downloaded

# Create video capture object
cap = cv.VideoCapture('./subject1/proefpersoon 1.2_M.avi')  # Replace with your video file path

# Optical flow parameters
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 20, 0.03))

# Check if video opened successfully
if not cap.isOpened():
    print("Unable to open video")
    exit()

# Open file for writing
with open("./Points/tongue_MiddleTracking_M1.txt", "w") as log_file:
    # log_file.write("Frame	Tongue_X	Tongue_Y\n")

    # Variables to control playback state
    paused = False
    tongue_visible = False
    tongue_point = None
    prev_tongue_point = None

    # Define distance thresholds
    mouth_open_threshold = 191  # Detect if mouth is open
    mouth_close_threshold = 170  # Detect if mouth is closed
    tongue_movement_threshold = 30  # Threshold for excessive tongue movement
    tongue_movement_min_threshold = 0.01  # Threshold for minimal tongue movement

    # Frame counter
    frame_counter = 0
    check_frames_interval = 20  # Check every 20 frames
    check_tracking = True  
    frame_cnt = 0

    # Function to manually select tongue tip point
    def select_tongue_point(frame):
        roi = cv.selectROI("Select Tongue Point", frame, fromCenter=False, showCrosshair=True)
        cv.destroyAllWindows()
        x = int(roi[0] + roi[2] / 2)
        y = int(roi[1] + roi[3] / 2)
        return np.array([[x, y]], dtype=np.float32).reshape(-1, 1, 2)

    # Start reading video
    while cap.isOpened():
        frame_cnt += 1
        if not paused:  # Only read next frame when not paused
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to grayscale, required by dlib face detector
            gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # Use dlib to detect faces
            faces = detector(gray_frame)

            # Loop over each detected face
            for face in faces:
                # Get 68 facial landmarks
                landmarks = predictor(gray_frame, face)

                # Draw each landmark point
                for i in range(68):
                    x = landmarks.part(i).x
                    y = landmarks.part(i).y
                    cv.circle(frame, (x, y), 2, (0, 255, 0), -1)  # Mark all landmarks in green

                # Get coordinates of landmark points 51 and 8
                x_51, y_51 = landmarks.part(51).x, landmarks.part(51).y  # Point 51
                x_8, y_8 = landmarks.part(8).x, landmarks.part(8).y  # Point 8

                # Calculate Euclidean distance between the two points
                distance = np.sqrt((x_51 - x_8) ** 2 + (y_51 - y_8) ** 2)

                # Display distance on video frame
                cv.putText(frame, f"Dist: {distance:.2f} Frame: {frame_cnt}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv.line(frame, (x_51, y_51), (x_8, y_8), (0, 0, 255), 2)

                # Determine if tongue tip needs manual selection
                if distance > mouth_open_threshold and not tongue_visible:
                    # Ask user if they can see the tongue tip
                    print("Mouth is open, can you see the tongue tip? (y/n)")
                    user_input = input().strip().lower()
                    if user_input == 'y':
                        # Pause video, manually select tongue tip point
                        paused = True
                        tongue_point = select_tongue_point(frame)  # Manually select tongue tip
                        tongue_visible = True
                        prev_tongue_point = tongue_point.copy()
                        paused = False

                # If tongue tip exists and mouth is closed, prompt user to clear tongue tip
                if tongue_visible and distance < mouth_close_threshold:
                    print("Mouth is closed, has the tongue tip disappeared? (y/n)")
                    user_input = input().strip().lower()
                    if user_input == 'y':
                        tongue_visible = False
                        tongue_point = None
                        # Ask user if they want to update threshold
                        print("Do you want to update the threshold? (y/n)")
                        user_input = input().strip().lower()
                        if user_input == 'y':
                            print("Please enter a new threshold:")  # Recommended update to 194
                            mouth_open_threshold = int(input().strip())
                    else:
                        # Ask user if they want to update close threshold
                        print("Do you want to update close threshold? (y/n)")
                        user_input = input().strip().lower()
                        if user_input == 'y':
                            print("Please enter a new threshold:")
                            mouth_close_threshold = int(input().strip())
    
                # If tongue tip exists, use optical flow to track tongue tip
                if tongue_visible and tongue_point is not None:
                    # Track tongue tip using optical flow
                    tongue_point_curr, st, err = cv.calcOpticalFlowPyrLK(prev_gray, gray_frame, tongue_point, None, **lk_params)

                    # Check if optical flow tracking is successful (st value of 1 indicates success)
                    if tongue_point_curr is not None and st[0][0] == 1:
                        tongue_point = tongue_point_curr  # Update tongue tip position
                        tongue_x, tongue_y = tongue_point.ravel()

                        # Check if tongue movement is excessive
                        tongue_movement = np.linalg.norm(tongue_point - prev_tongue_point)
                        if tongue_movement > tongue_movement_threshold:
                            # If tongue movement is excessive, pause video and prompt user
                            paused = True
                            print("Excessive tongue movement, was there an error? (y/n)")
                            user_input = input().strip().lower()
                            if user_input == 'y':
                                # User confirms error, reselect tongue tip
                                tongue_point = select_tongue_point(frame)
                            prev_tongue_point = tongue_point.copy()
                            paused = False
                        elif tongue_movement <= tongue_movement_min_threshold:
                            # If tongue movement is minimal, pause video and prompt user
                            paused = True
                            print("Minimal tongue movement, was there an error? (y/n)")
                            user_input = input().strip().lower()
                            if user_input == 'y':
                                # User confirms error, reselect tongue tip
                                tongue_point = select_tongue_point(frame)
                            prev_tongue_point = tongue_point.copy()
                            paused = False
                        else:
                            prev_tongue_point = tongue_point.copy()
                        
                        # Write current frame number and tongue tip coordinates to file
                        log_file.write(f"{frame_cnt} {int(tongue_x)} {int(tongue_y)}\n")

                        # Check tracking point accuracy every 20 frames
                        if check_tracking:
                            frame_counter += 1
                            if frame_counter >= check_frames_interval:
                                paused = True
                                print("Is the tracking point correct? (y/n)")
                                user_input = input().strip().lower()
                                if user_input == 'n':
                                    # User confirms error, reselect tongue tip
                                    tongue_point = select_tongue_point(frame)
                                elif user_input == 'y':
                                    # User confirms correct tracking, stop checking every 20 frames
                                    check_tracking = False
                                frame_counter = 0  # Reset frame counter
                                paused = False

                        cv.circle(frame, (int(tongue_x), int(tongue_y)), 5, (0, 0, 255), -1)  # Mark tongue tip in red

        # Display each processed frame
        cv.imshow('Face and Tongue Tracking', frame)

        # Check keyboard input
        key = cv.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):  # Press 'p' to pause
            paused = True
            print("Is there an error with the tracking point? (y/n)")
            user_input = input().strip().lower()
            if user_input == 'y':
                # User confirms error, reselect tongue tip
                tongue_point = select_tongue_point(frame)
                paused = False
        elif key == ord('c'):  # Press 'c' to continue
            paused = False

        prev_gray = gray_frame.copy()  # Update previous frame's grayscale image

# Release video capture object and close all windows
cap.release()
cv.destroyAllWindows()
