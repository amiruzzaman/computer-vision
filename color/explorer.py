import cv2
import numpy as np

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    print("Press 'q' to quit the application.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 2. ISOLATE SHADOWS (Low Value)
        # Goal: Find pixels where brightness is between 0 and 50
        lower_shadow = np.array([0, 0, 0])
        upper_shadow = np.array([179, 255, 60])
        shadow_mask = cv2.inRange(hsv, lower_shadow, upper_shadow)

        # 3. ISOLATE HIGHLIGHTS (High Value)
        # Goal: Find pixels where brightness is very high
        lower_light = np.array([0, 0, 200])
        upper_light = np.array([179, 80, 255])
        light_mask = cv2.inRange(hsv, lower_light, upper_light)

        # 4. STUDENT TASK: COLOR TRACKING
        # Choose a vibrant color (e.g., Blue, Green, or Red)
        # Define the lower and upper HSV boundaries for that color
        # lower_color = np.array([H_min, S_min, V_min])
        # upper_color = np.array([H_max, S_max, V_max])
        # color_mask = cv2.inRange(hsv, lower_color, upper_color)

        # 5. VISUALIZATION
        # Create a copy to draw on
        output = frame.copy()
        
        # Colorize the shadows purple and highlights yellow in the output
        output[shadow_mask > 0] = [255, 0, 150] # BGR for Purple
        output[light_mask > 0] = [0, 255, 255]  # BGR for Yellow

        # Show the result
        cv2.imshow('Light, Shadow, and Color Explorer', output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()