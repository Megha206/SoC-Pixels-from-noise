import cv2
import numpy as np

# Open webcam
vid = cv2.VideoCapture(0)

while True:
    # Capture frame
    _, frame = vid.read()


    # Get frame dimensions
    height, width, _ = frame.shape

    # Define ROI (center 1/3rd of the frame)
    top = height // 3 
    bottom = 2 * height // 3
    left = width // 3  
    right = 2 * width // 3
    roi = frame[top:bottom, left:right]

 

    # Compute mean colors in ROI
    b_mean = np.mean(roi[:, :, 0])
    g_mean = np.mean(roi[:, :, 1])
    r_mean = np.mean(roi[:, :, 2])

    # Determine dominant color
    if b_mean > g_mean and b_mean > r_mean:
        color_name = "Blue"
        color_bgr = (255, 0, 0)
    elif g_mean > r_mean and g_mean > b_mean:
        color_name = "Green"
        color_bgr = (0, 255, 0)
    else:
        color_name = "Red"
        color_bgr = (0, 0, 255)

    # Draw ROI rectangle on frame
    cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 255), 2)

    # Display the color on the frame
    cv2.putText(frame, f"Dominant: {color_name}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color_bgr, 2)

    # Show the frame
    cv2.imshow("frame", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
vid.release()
cv2.destroyAllWindows()


