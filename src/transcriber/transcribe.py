import pytesseract
from moviepy.video.io.VideoFileClip import VideoFileClip
import whisper
import os
from transcriber import ocr_extraction
from transcriber.video_download import download_video_by_id

# Konfiguration für Tesseract (Pfad zur ausführbaren Datei anpassen)
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"  # Passe den Pfad an

# Funktion zum Teilen des Videos in Abschnitte
def segment_video(delegate_id, video_file, ocr_timestamps):
    video = VideoFileClip(video_file)
    segment_files = []

    for i in range(len(ocr_timestamps)):
        start_time = ocr_timestamps[i]
        end_time = ocr_timestamps[i + 1] if i + 1 < len(ocr_timestamps) else video.duration
        segment = video.subclipped(start_time, end_time)
        segment_file = f"segment_{delegate_id}_{i + 1}.mp4"
        segment.write_videofile(segment_file, codec="libx264", audio_codec="aac")
        segment_files.append(segment_file)

    return segment_files

#model = whisper.load_model("medium")

# Funktion zum Transkribieren von Videoabschnitten
def transcribe_segments(segment_files, model):
    transcriptions = []

    for segment_file in segment_files:
        print(f"Transkribiere {segment_file}...")
        result = model.transcribe(segment_file, language="de")
        transcriptions.append(result["text"].strip())

    return transcriptions

# Funktion zum Kombinieren von OCR- und Transkriptionsergebnissen
def combine_ocr_and_transcriptions(ocr_results, transcriptions):
    combined_results = []

    for i, (timestamp, question) in enumerate(ocr_results):
        answer = transcriptions[i] if i < len(transcriptions) else ""
        combined_results.append({
            "timestamp": timestamp,
            "question": question,
            "answer": answer
        })

    return combined_results

# Funktion zum Verarbeiten eines Videos und Speichern der Ergebnisse in der DB
def process_video(delegate_id, model, directory="delegate_videos"):
    video_file = os.path.join(directory, f"{delegate_id}.mp4")

    if not os.path.exists(video_file):
        video_file = download_video_by_id(delegate_id, directory)

    # Schritt 2: OCR-Ergebnisse extrahieren
    questions, ocr_timestamps = ocr_extraction.extract_blue_ocrs_with_timestamps(video_file)

    try:
        # Schritt 3: Video in Abschnitte teilen
        segment_files = segment_video(delegate_id, video_file, ocr_timestamps)
    except:
        print(f"CANNOT SEGMENT VIDEO {video_file}")
        return

    # Schritt 4: Segmente transkribieren
    transcriptions = transcribe_segments(segment_files, model)

    # Schritt 5: Ergebnisse kombinieren und in die DB einfügen
    combined_results = combine_ocr_and_transcriptions(list(zip(ocr_timestamps, questions)), transcriptions)
    for segment_file in segment_files:
        os.remove(segment_file)
    
    return combined_results


if __name__ == "__main__":
    video_directory = "delegate_videos"  
    delegate_ids = [5678, 30670, 83153]  
    model = whisper.load_model(
        name="medium",
        device="cpu")

    for delegate_id in delegate_ids:
        process_video(delegate_id, model, video_directory)
    # process_all_videos_in_directory(video_directory)
