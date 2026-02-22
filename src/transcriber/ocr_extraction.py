from transcriber import frame_check
import pytesseract
import cv2

def extract_blue_ocrs_with_timestamps(video_file):
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Fehler: Video konnte nicht geöffnet werden.")
        return [], []

    extracted_texts = []
    timestamps = []
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0

    blue_encountered = False

    stored_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1;
        is_blue = frame_check.check_for_blue(frame)
        if is_blue and not blue_encountered:
            blue_encountered = True
        if is_blue and blue_encountered:
            stored_frames.append((frame_count, frame))
        
        if blue_encountered and not is_blue:
            blue_encountered = False

            frame_num, ocr_frame = stored_frames[len(stored_frames) // 2]
            gray = cv2.cvtColor(ocr_frame, cv2.COLOR_BGR2GRAY)
            _, gray = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
            text = pytesseract.image_to_string(gray).strip()
            if text:  # Nur neuen Text speichern
                extracted_texts.append(text)
                timestamps.append(frame_num / frame_rate)

            stored_frames = []
            
            
    for i, (text, timestamp) in enumerate(zip(extracted_texts, timestamps)):
        print(f"[{timestamp:.2f}s] OCR-Ergebnis {i + 1}: {text}")

    return extracted_texts, timestamps


