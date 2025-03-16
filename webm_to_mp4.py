import ffmpeg

def convert_webm_to_mp4(input_file, output_file):
    try:
        process = (
            ffmpeg
            .input(input_file)
            .output(output_file, vcodec='libx264')
            .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
        )
        print("FFmpeg Output:", process)
        print(f"Conversion successful: {output_file}")
    except ffmpeg.Error as e:
        print("FFmpeg Error:", e.stderr.decode())

# Example usage
convert_webm_to_mp4("./audio/BgNdtk9h8Ok.webm", "./audio/BgNdtk9h8Ok.mp4")
