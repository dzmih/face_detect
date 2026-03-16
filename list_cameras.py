import cv2

def list_cameras():
    # check first 5 indices
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"camera found at index: {i}")
            cap.release()

list_cameras()