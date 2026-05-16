import cv2
import os
import re
import pandas as pd
from datetime import datetime, timedelta
from deepface import DeepFace
import time

# ================= PATH SETTINGS =================
ROUTINE_FILE = r"C:\Users\Rupsha Biswas\Pictures\face detection\CSE_class_routine.xlsx"
IMAGES_PATH = r"C:\Users\Rupsha Biswas\Pictures\face detection\images_data"
ATTENDANCE_FOLDER = r"C:\Users\Rupsha Biswas\Pictures\face detection\attendance_sheets"

# Attendance timing controls
START_DELAY_MIN = 5      # webcam starts 5 minutes after class start
SESSION_DURATION_MIN = 10
IDLE_SLEEP_SEC = 20

# ================= CAMERA INDEX DETECTION =================
def find_camera_index():
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    for idx in range(5):  # try 0-4
        for be in backends:
            cap = cv2.VideoCapture(idx, be)
            if cap.isOpened():
                cap.release()
                print(f"[INFO] ✅ Camera found at index {idx} with backend {be}")
                return idx
    print("[ERROR] ❌ No camera found! Check permissions or drivers.")
    return None

CAMERA_INDEX = find_camera_index()
if CAMERA_INDEX is None:
    raise SystemExit

# ================= LOAD KNOWN FACES =================
print("[INFO] Loading known faces...")
known_faces = {}
if os.path.isdir(IMAGES_PATH):
    for img_name in os.listdir(IMAGES_PATH):
        img_path = os.path.join(IMAGES_PATH, img_name)
        if os.path.isfile(img_path):
            name = os.path.splitext(img_name)[0]
            known_faces[name] = img_path
print(f"[INFO] Loaded {len(known_faces)} known faces.")

# ================= LOAD ROUTINE =================
if not os.path.exists(ROUTINE_FILE):
    raise FileNotFoundError(f"Routine file not found at: {ROUTINE_FILE}")

routine_df = pd.read_excel(ROUTINE_FILE)
print("[DEBUG] Routine columns:", routine_df.columns.tolist())

# ================= HELPERS =================
DAY_MAP = {
    "mon": "monday", "monday": "monday",
    "tue": "tuesday", "tues": "tuesday", "tuesday": "tuesday",
    "wed": "wednesday", "weds": "wednesday", "wednesday": "wednesday",
    "thu": "thursday", "thur": "thursday", "thurs": "thursday", "thursday": "thursday",
    "fri": "friday", "friday": "friday",
    "sat": "saturday", "saturday": "saturday",
    "sun": "sunday", "sunday": "sunday",
    "all": "all", "daily": "all", "everyday": "all"
}

def normalize_day_token(token: str):
    if not token:
        return None
    t = token.strip().lower()
    t = re.sub(r'[^a-z]', '', t)  # keep letters only
    return DAY_MAP.get(t, t)

def parse_days_cell(value):
    """
    Accepts values like:
    - 'Mon, Wed, Fri'
    - 'Monday/Thursday'
    - 'Tuesday & Thursday'
    - 'Daily', 'All'
    Returns a set of normalized day names.
    """
    if pd.isna(value):
        return set()
    s = str(value).strip()
    if not s:
        return set()

    # Quick check for daily/all
    if s.lower() in ("all", "daily", "everyday"):
        return {"all"}

    # Split on commas, slashes, ampersands, 'and'
    parts = re.split(r'[,\-/&]| and ', s, flags=re.IGNORECASE)
    days = set()
    for p in parts:
        norm = normalize_day_token(p)
        if norm:
            if norm == "all":
                return {"all"}
            days.add(norm)
    return days

def normalize_day(day_val):
    if pd.isna(day_val):
        return None
    return normalize_day_token(str(day_val))

def parse_time_cell(value):
    if pd.isna(value):
        return None

    # Already datetime/time-like
    if hasattr(value, "hour") and hasattr(value, "minute"):
        try:
            return value.to_pydatetime().time()
        except Exception:
            try:
                return value.time()
            except Exception:
                pass

    # Excel serial (float or int)
    if isinstance(value, (int, float)):
        try:
            base = datetime(1899, 12, 30)  # Excel base
            return (base + timedelta(days=float(value))).time()
        except Exception:
            pass

    # Strings: also tolerate ranges like "10:00 AM - 11:00 AM"
    s = str(value).strip()

    # If it's a range, try to take the first time token
    # Matches: 10, 10:00, 10:00 AM, 22:15, etc.
    m = re.search(r'(\d{1,2}(:\d{2})?\s*(AM|PM|am|pm)?)', s)
    if m:
        candidate = m.group(1)
        for fmt in ("%I:%M %p", "%I %p", "%H:%M", "%H:%M:%S", "%I:%M:%S %p"):
            try:
                return datetime.strptime(candidate.upper(), fmt).time()
            except Exception:
                continue

    # Try full-string parses
    for fmt in ("%H:%M:%S", "%H:%M", "%I:%M %p", "%I:%M:%S %p", "%I %p"):
        try:
            return datetime.strptime(s.upper(), fmt).time()
        except Exception:
            continue

    return None

def auto_detect_columns(df):
    colmap = {"day": None, "time": None, "subject": None, "end_time": None}
    for col in df.columns:
        c = col.strip().lower()
        if "day" in c:
            colmap["day"] = col
        elif "start" in c and "time" in c:
            colmap["time"] = col
        elif c == "time":  # fallback
            colmap["time"] = col
        elif ("subject" in c) or ("class" in c) or ("course" in c):
            colmap["subject"] = col
        elif ("end" in c and "time" in c) or c == "end time":
            colmap["end_time"] = col
    return colmap

colmap = auto_detect_columns(routine_df)
print("[DEBUG] Detected column mapping:", colmap)

def get_subject_from_row(row):
    subj_col = colmap["subject"]
    if subj_col and pd.notna(row[subj_col]):
        return str(row[subj_col]).strip()
    return "Unknown"

def is_class_today(row, today_norm):
    day_col = colmap["day"]
    if not day_col:
        return False
    day_set = parse_days_cell(row[day_col])
    if "all" in day_set:
        return True
    return today_norm in day_set

def get_today_classes_parsed():
    """Returns list of (subject, start_time) for today's classes with parsed times, sorted by time."""
    now = datetime.now()
    today_norm = normalize_day(now.strftime("%A"))
    if not colmap["day"] or not colmap["time"]:
        print("[ERROR] Could not detect 'Day' or 'Start Time' columns in routine.")
        return []

    # Filter rows where today's day is present in the 'Days' cell (supports multi-day cells)
    today_rows = routine_df[routine_df.apply(lambda r: is_class_today(r, today_norm), axis=1)]
    if today_rows.empty:
        print(f"[DEBUG] No classes found in Excel for {today_norm}")
        return []

    parsed = []
    for _, row in today_rows.iterrows():
        t = parse_time_cell(row[colmap["time"]])
        if t:
            parsed.append((get_subject_from_row(row), t))
        else:
            print(f"[WARN] Could not parse Start Time from value: {row[colmap['time']]}")

    parsed.sort(key=lambda x: x[1])
    return parsed

def get_current_class():
    now = datetime.now()
    parsed_classes = get_today_classes_parsed()

    if not parsed_classes:
        return None

    print(f"[DEBUG] Classes for {now.strftime('%A').lower()}:")
    for subj, t in parsed_classes:
        print(f"   - {subj} at {t.strftime('%H:%M')}")

    for subj, t in parsed_classes:
        class_dt = datetime.combine(now.date(), t)
        start_window = class_dt + timedelta(minutes=START_DELAY_MIN)
        end_window = start_window + timedelta(minutes=SESSION_DURATION_MIN)
        print(f"[DEBUG] Check '{subj}': start={class_dt.time()} | "
              f"active={start_window.time()}..{end_window.time()} | now={now.time()}")
        if start_window <= now <= end_window:
            return subj

    # Not inside any window — show next upcoming
    future = [(s, t) for s, t in parsed_classes if datetime.combine(now.date(), t) > now]
    if future:
        s, t = future[0]
        print(f"[INFO] Next upcoming class: {s} at {t.strftime('%H:%M')}")
    else:
        print("[INFO] No more classes for today.")
    return None

# ================= ATTENDANCE STORAGE =================
def mark_attendance(subject, present_names):
    date_str = datetime.now().strftime("%Y-%m-%d")
    folder_path = os.path.join(ATTENDANCE_FOLDER, date_str)
    os.makedirs(folder_path, exist_ok=True)

    file_path = os.path.join(folder_path, f"{subject}.xlsx")

    if os.path.exists(file_path):
        df = pd.read_excel(file_path)
        if date_str not in df.columns:
            df[date_str] = 0
    else:
        df = pd.DataFrame(columns=["Name", "Roll No", date_str])

    for name in present_names:
        roll_no = name.split("_")[-1] if "_" in name else ""
        if name not in df["Name"].values:
            new_row = {"Name": name, "Roll No": roll_no, date_str: 1}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        else:
            df.loc[df["Name"] == name, date_str] = 1

    df.to_excel(file_path, index=False)
    print(f"[INFO] Attendance saved for {subject}.")

# ================= FACE RECOGNITION (CORRECTED) =================
def recognize_faces(camera_index=0):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index, cv2.CAP_MSMF)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"[ERROR] Webcam index {camera_index} could not be opened! Check drivers/permissions.")
        return set()

    print(f"[INFO] Webcam opened at index {camera_index}")
    present_names = set()
    start_time = time.time()

    # --- DeepFace Database Warm-up (Optional, but recommended for speed) ---
    # This call creates a single, cached representation of your IMAGES_PATH folder
    # to speed up subsequent searches inside the loop.
    try:
        if os.path.isdir(IMAGES_PATH) and os.listdir(IMAGES_PATH):
            print("[INFO] Creating database representation for quick search...")
            DeepFace.find(
                img_path=os.path.join(IMAGES_PATH, os.listdir(IMAGES_PATH)[0]),
                db_path=IMAGES_PATH,
                model_name="ArcFace",
                enforce_detection=False,
                silent=True
            )
    except Exception as e:
        print(f"[WARN] DeepFace database warm-up failed (this is often fine if no faces are registered): {e}")
    # -----------------------------------------------------------------------

    while (time.time() - start_time) < SESSION_DURATION_MIN * 60:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame from webcam.")
            break

        # Flip the frame for a more natural mirror-like view
        frame = cv2.flip(frame, 1)

        try:
            # Use DeepFace.find to get face detections and identities
            # DeepFace.find returns a list of dataframes for each detected face
            results = DeepFace.find(
                img_path=frame,
                db_path=IMAGES_PATH,
                model_name="ArcFace",
                enforce_detection=False,
                silent=True # Suppress deepface logging in the loop
            )
        except Exception:
            # This is a common occurrence when no face is detected in the frame
            results = []

        if results and isinstance(results, list):
            for result_df in results:
                if not result_df.empty:
                    # Get the bounding box coordinates (the detected face region)
                    x = int(result_df['source_x'][0])
                    y = int(result_df['source_y'][0])
                    w = int(result_df['source_w'][0])
                    h = int(result_df['source_h'][0])
                    
                    # Get the recognized identity and name
                    identity_path = result_df.iloc[0]["identity"]
                    name = os.path.splitext(os.path.basename(identity_path))[0]
                    
                    # Add to present list
                    present_names.add(name)

                    # Extract the Roll No. (assuming format 'Name_RollNo')
                    roll_no = name.split("_")[-1] if "_" in name else "N/A"
                    
                    # Determine display text and confidence (optional: use result_df.iloc[0]["VGG-Face_cosine"] for confidence)
                    display_text = f"{name}"
                    roll_text = f"Roll: {roll_no}"

                    # --- DRAWING ---
                    # Draw the bounding box (grid box)
                    color = (0, 255, 0) # Green
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                    # Draw background rectangle for text
                    text_y_start = y - 30
                    if text_y_start < 0: text_y_start = y + h + 10 # Place below if box is too high

                    cv2.rectangle(frame, (x, text_y_start), (x + 250, text_y_start + 50), color, -1) # Solid green background

                    # Put the name on the frame
                    cv2.putText(
                        frame,
                        display_text,
                        (x + 5, text_y_start + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 0), # Black text
                        2,
                    )
                    
                    # Put the roll number on the frame
                    cv2.putText(
                        frame,
                        roll_text,
                        (x + 5, text_y_start + 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0), # Black text
                        2,
                    )
                    # ---------------

        # Display the frame in a window
        cv2.imshow("Attendance System - Press 'q' to quit", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and destroy all windows
    cap.release()
    cv2.destroyAllWindows()
    return present_names
# ================= MAIN LOOP =================
if __name__ == "__main__":
    TEST_MODE = False  # Set True to force webcam test immediately (ignores schedule)

    if TEST_MODE:
        print("[TEST] Forcing webcam open for test…")
        names = recognize_faces(camera_index=CAMERA_INDEX)
        print("[TEST] Detected:", names)
    else:
        print("[INFO] Attendance system running…")
        while True:
            subject = get_current_class()
            if subject:
                print(f"[INFO] Class detected: {subject}. Starting attendance…")
                names = recognize_faces(camera_index=CAMERA_INDEX)
                mark_attendance(subject, names)
                print("[INFO] Waiting for next class…")
                time.sleep(60)
            else:
                print("[DEBUG] No class window right now. Sleeping", IDLE_SLEEP_SEC, "s.")
                time.sleep(IDLE_SLEEP_SEC)
