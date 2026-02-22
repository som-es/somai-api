import cv2

SQUARE_TO_CHECK=32
THRESHOLD=40
TARGET_COLOR1=(23, 39, 63)
TARGET_COLOR2=(11, 25, 44)

def check_pixel(target, pixel) -> bool:
    return abs(target[0] - int(pixel[2])) + \
                abs(target[1] - int(pixel[1])) + \
                abs(target[2] - int(pixel[0])) > THRESHOLD

def check_square(frame, row_offset, col_offset):
    for row in range(row_offset, row_offset + SQUARE_TO_CHECK):
        for col in range(col_offset, col_offset + SQUARE_TO_CHECK):
            pixel = frame[row, col]
            if check_pixel(TARGET_COLOR1, pixel) and check_pixel(TARGET_COLOR2, pixel):
                return False

    return True

def check_for_blue(frame,):
    return check_square(frame, 0, 0) and \
        check_square(frame, frame.shape[0]-SQUARE_TO_CHECK, frame.shape[1]-SQUARE_TO_CHECK)

if __name__ == "__main__":
    video_file = "AbgeordnetenVideos/35520.mp4"
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Fehler: Video konnte nicht geöffnet werden.")
        exit()

    extracted_texts = []
    timestamps = []
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        print(check_for_blue(frame))
        cv2.imshow("nix", frame)
        cv2.waitKey(0)