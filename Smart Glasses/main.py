import cv2
import os
import re
from ultralytics import YOLO
from deepface import DeepFace

class CameraFeed:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.frame = None
        self.capture_flag = False
        self.function_choice = 0
        self.output_folder = r".\captured_frames"  # Folder to store captured frames
        self.image_path = r".\captured_frames\frame.jpg"

        # Create the output folder if it doesn't exist
        os.makedirs(self.output_folder, exist_ok=True)

    def start(self):
        while True:
            ret, frame = self.cap.read()
            self.frame = frame  # Store the captured frame in the 'frame' variable

            if self.capture_flag:
                self.save_captured_frame()
                if self.function_choice == 1:
                    self.recognize_face()
                elif self.function_choice == 2:
                    self.recgonize_object()
                self.capture_flag = False

            cv2.imshow('Camera Feed', frame)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            elif key & 0xFF == ord('f'):
                self.capture_flag = True
                self.function_choice = 1
            elif key & 0xFF == ord('o'):
                self.capture_flag = True
                self.function_choice = 2

        self.cap.release()
        cv2.destroyAllWindows()

    def save_captured_frame(self):
        if self.frame is not None:
            # Generate a unique filename
            filename = os.path.join(self.output_folder, "frame.jpg")

            # Save the captured frame as an image
            cv2.imwrite(filename, self.frame)
            print(f"Captured frame saved as {filename}")
        else:
            print("No frame captured.")

    def recgonize_object(self):
        model = YOLO("yolov8x.pt")
        model(self.image_path)

    def recognize_face(self):
        recongnized_emotion = DeepFace.analyze(img_path=self.image_path, enforce_detection=False)
        extracted_info = {
        'dominant_emotion': recongnized_emotion[0]['dominant_emotion'],
        'age': recongnized_emotion[0]['age'],
        'dominant_gender': recongnized_emotion[0]['dominant_gender'],
        'dominant_race': recongnized_emotion[0]['dominant_race']}
        for key, value in extracted_info.items():
            print(f"{key}: {value}")
    
if __name__ == '__main__':
    feed = CameraFeed()
    feed.start()




    
    


