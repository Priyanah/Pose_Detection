import cv2
import math as m
import mediapipe as mp
from plyer import notification

# offset distance
def finddistance(x1, y1, x2, y2):
    return m.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# calculate body posture
def findangle(x1, y1, x2, y2):
    angle = m.degrees(m.acos((y2 - y1)*(-y1) / (m.sqrt((x2 - x1)**2 + (y2 - y1)**2) * y1)))
    return angle

# send warning
def poseWarning(x):
    notification.notify(
        title="Posture Alert",
        message=f"Poor posture detected! Angle: {x} degrees",
        timeout=10
    )


# Colors and font
font = cv2.FONT_HERSHEY_SIMPLEX
green = (127, 255, 0)
red = (50, 50, 255)
yellow = (0, 255, 255)
pink = (255, 0, 255)

# Thresholds for posture
NECK_THRESHOLD = 25  # Neck angle threshold for "bad" posture
TORSO_THRESHOLD = 30  # Torso angle threshold for "bad" posture
SHOULDER_OFFSET_THRESHOLD = 100  # Shoulder offset for alignment

# Mediapipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Video file and capture
filename = 'Test.mp4'
cap = cv2.VideoCapture(filename)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_size = (width, height)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_output = cv2.VideoWriter('Tested.mp4', fourcc, fps, frame_size)

print(f"Width : {width}")
print(f"height : {height}")

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
else:
    print(f"Video opened successfully: {cap}")

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Reached the end of the video.")
            break

        # Convert the BGR image to RGB for Mediapipe processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        keypoints = pose.process(image_rgb)
        lm = keypoints.pose_landmarks
 
        if lm:  # If landmarks are detected
            lmPose = mp_pose.PoseLandmark

            # Extract key points
            l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * width)
            l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * height)
            r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * width)   # to calculate offset distance
            r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * height)  # to calculate offset distance
            l_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * width)
            l_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * height)
            l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * width)
            l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * height)

            # Calculate offset (alignment) and angles
            offset = finddistance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)
            neck_inclination = findangle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
            torso_inclination = findangle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)

            # Draw landmarks
            cv2.circle(image, (l_shldr_x, l_shldr_y), 7, yellow, -1)
            cv2.circle(image, (l_ear_x, l_ear_y), 7, yellow, -1)
            cv2.circle(image, (r_shldr_x, r_shldr_y), 7, pink, -1)
            cv2.circle(image, (l_hip_x, l_hip_y), 7, yellow, -1)

            # Check posture status
            posture_status = "Good Posture"
            color = green

           # After calculating angles
            if neck_inclination > NECK_THRESHOLD or torso_inclination > TORSO_THRESHOLD or offset > SHOULDER_OFFSET_THRESHOLD:
                posture_status = "Bad Posture"
                color = red
                poseWarning(neck_inclination)  # Trigger a warning notification
            else:
                posture_status = "Good Posture"
                color = green
                

            # Show posture status
            cv2.putText(image, f"Posture: {posture_status}", (30, 30), font, 0.9, color, 2)

            # Show angles and offset
            angle_text = f"Neck: {int(neck_inclination)} | Torso: {int(torso_inclination)}"
            cv2.putText(image, angle_text, (30, 60), font, 0.7, color, 2)

            if offset < SHOULDER_OFFSET_THRESHOLD:
                cv2.putText(image, f"Aligned: {int(offset)}", (width - 150, 30), font, 0.9, green, 2)
            else:
                cv2.putText(image, f"Not Aligned: {int(offset)}", (width - 150, 30), font, 0.9, red, 2)

        # Display the processed frame
        cv2.imshow("Posture Analysis", image)

        # Write the frame to the output video
        video_output.write(image)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    video_output.release()
    cv2.destroyAllWindows()
