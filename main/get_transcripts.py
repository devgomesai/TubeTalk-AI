import os
import assemblyai as aai
from dotenv import load_dotenv
from urllib.parse import urlparse, parse_qs
import yt_dlp
load_dotenv()

aai.settings.api_key = os.getenv("ASSEMBLY_API_API")

def get_video_id(url):
    parsed_url = urlparse(url)
    if parsed_url.hostname in ["www.youtube.com", "youtube.com"]:
        if parsed_url.path.startswith("/live/"):
            return parsed_url.path.split("/live/")[1]
        return parse_qs(parsed_url.query).get("v", [None])[0]
    elif parsed_url.hostname == "youtu.be":
        return parsed_url.path[1:]
    return None

def download_audio(youtube_url):
    print("Downloading audio from YouTube for transcription...")
    try:
        video_id = get_video_id(youtube_url)
        if not video_id:
            print("Invalid YouTube URL")
            return "Invalid YouTube URL", None
        audio_dir = "audio"
        os.makedirs(audio_dir, exist_ok=True)
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": f"{audio_dir}/{video_id}.%(ext)s",
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            print("Audio download complete")
            return f"{audio_dir}/{video_id}.{info['ext']}", info.get("title", "Unknown Title")
    except Exception as e:
        print(f"Audio download failed: {str(e)}")
        return f"Audio download failed: {str(e)}", None
    
def get_transcription(video_path: str):
    if not aai.settings.api_key:
        return "AssemblyAI API key is required for transcription", None

    if not os.path.exists(video_path):
        return "Invalid video path", None

    # Create transcripts folder if not exists
    transcripts_dir = os.path.join(os.getcwd(), "transcripts")
    os.makedirs(transcripts_dir, exist_ok=True)

    # Use filename (without extension) as transcript name
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    transcript_path = os.path.join(transcripts_dir, f"{video_id}.txt")

    # If transcript already exists, load it
    if os.path.exists(transcript_path):
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript_text = f.read()
        return transcript_text, video_id

    # Transcribe the given video/audio file
    transcriber = aai.Transcriber()
    config = aai.TranscriptionConfig(
        speaker_labels=True,
        punctuate=True,
        format_text=True
    )
    transcript_obj = transcriber.transcribe(video_path, config)

    if transcript_obj.text:
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(transcript_obj.text)
        print(f"Transcription saved at: {transcript_path}")
        return transcript_obj.text, video_id

    return "Failed to generate transcript", None

if __name__ == '__main__':
    # Example usage
    # "https://www.youtube.com/watch?v=H3KnMyojEQU"
    youtube_url = input("Enter a Video URL: ")
    video_path, title = download_audio(youtube_url)
    get_transcription(video_path)

