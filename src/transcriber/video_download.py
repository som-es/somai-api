
import requests
import os

def download_video_by_id(delegate_id: int, directory: str) -> str:
    """
    Downloads a delegate's video.
    Returns the filepath on success. Raises an exception on failure.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    url = f"https://www.parlament.gv.at/person/{delegate_id}?json=True"

    # Fetch JSON data
    response = requests.get(url, timeout=10)
    response.raise_for_status()  
    
    data = response.json() 

    # Extract the internal ID (pad_intern) and video ID
    biografie = data.get('content', {}).get('biografie', {})
    pad_intern = data.get('content', {}).get('personInfo', {}).get('pad_intern')
    indirect_video_id = biografie.get('portraitvideo')

    if not indirect_video_id or not pad_intern:
        raise ValueError(f"No video or internal ID found for delegate ID {delegate_id}.")

    indirect_video_url = f"https://api.ausp.cloud.insysgo.com/media/clean/{indirect_video_id}.mp4"

    # Fetch video URL metadata
    video_response = requests.get(indirect_video_url, timeout=10)
    video_response.raise_for_status()
    
    data_video = video_response.json()
    video_URL = data_video.get('URL')

    if not video_URL:
        raise ValueError("No URL found in the video metadata.")

    filename = f"{pad_intern}.mp4"
    filepath = os.path.join(directory, filename)

    # Download actual video file
    download_response = requests.get(video_URL, stream=True, timeout=20)
    download_response.raise_for_status()

    with open(filepath, 'wb') as file:
        for chunk in download_response.iter_content(chunk_size=8192):
            file.write(chunk)

    return filepath

def download_videos(directory="delegate_videos"):
    # Create directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    delegate_ids = [5678, 30670, 83153]

    for delegate_id in delegate_ids:
        try:
            download_video_by_id(delegate_id, directory)
        except Exception as e:
            print(f"Error downloading video for delegate ID {delegate_id}: {e}")


if __name__ == "__main__":
    download_videos()
