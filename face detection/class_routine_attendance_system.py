import cv2
import os
import pandas as pd
import face_recognition
import numpy as np
import datetime
import time
from deepface import DeepFace

# Load Excel routine
routine_df = pd.read_excel(r"C:\Users\Rupsha Biswas\Pictures\images_data\CSE_Class_Routine.xlsx")
print(routine_df.dtypes)

# Load known face data
print("[INFO] Loading known face images from images_data folder...")
known_face_encodings = []
known_face_names = []

images_path = r"C:\Users\Rupsha Biswas\Pictures\images_data"
for filename in os.listdir(images_path):
    if filename.lower().endswith((".jpg", ".png")):
        image = face_recognition.load_image_file(os.path.join(images_path, filename))
        encoding = face_recognition.face_encodings(image)
        if encoding:
            known_face_encodings.append(encoding[0])
            name_parts = os.path.splitext(filename)[0].split("_")
            if len(name_parts) >= 2:
                known_face_names.append((name_parts[0], name_parts[1]))
            else:
                known_face_names.append((filename, "Unknown"))

# Attendance marking function
def mark_attendance(subject, name, roll):
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.datetime.now().strftime("%H:%M:%S")
    attendance_file = "attendance.csv"
    if not os.path.exists(attendance_file):
        with open(attendance_file, "w") as f:
            f.write("Date,Subject,Name,Roll,Time\n")
    with open(attendance_file, "a") as f:
        f.write(f"{date_str},{subject},{name},{roll},{time_str}\n")

# Track processed classes per day
processed_classes = set()

# Continuous Monitoring
print("[INFO] Starting attendance monitoring loop...")
while True:
    now = datetime.datetime.now()
    current_day = now.strftime("%A")
    current_time = now.time()
    today_str = now.strftime("%Y-%m-%d")

    for index, row in routine_df.iterrows():
        subject = row["Class"]
        department = row["Department"]
        day = row["Days"]
        start_time = row["Start Time"]

        # Parse time strings if needed
        if isinstance(start_time, str):
            start_time = datetime.datetime.strptime(start_time, "%H:%M:%S").time()

        webcam_start = (datetime.datetime.combine(datetime.date.today(), start_time) + datetime.timedelta(minutes=5)).time()
        webcam_end = (datetime.datetime.combine(datetime.date.today(), start_time) + datetime.timedelta(minutes=15)).time()

        # Skip if already processed
        unique_class_key = f"{today_str}_{subject}"
        if unique_class_key in processed_classes:
            continue

        if current_day.lower() == day.lower() and webcam_start <= current_time <= webcam_end:
            print(f"[INFO] {subject} class is running. Starting webcam for attendance...")
            cap = cv2.VideoCapture(0)
            face_names_recorded = set()
            session_start = time.time()

            while time.time() - session_start < 600:  # 10 min session
                ret, frame = cap.read()
                if not ret:
                    break
                rgb_frame = frame[:, :, ::-1]
                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                for face_encoding, face_location in zip(face_encodings, face_locations):
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)

                    if matches[best_match_index]:
                        name, roll = known_face_names[best_match_index]
                        identity = f"{name} ({roll})"
                        if identity not in face_names_recorded:
                            mark_attendance(subject, name, roll)
                            face_names_recorded.add(identity)
                            print(f"[INFO] Marked Present: {identity}")

                        top, right, bottom, left = face_location
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv2.putText(frame, identity, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                cv2.imshow("Attendance System", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            cap.release()
            cv2.destroyAllWindows()
            processed_classes.add(unique_class_key)  # Mark this class done

    time.sleep(30)  # Check every 30 seconds
