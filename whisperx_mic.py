import pyaudio
import whisperx

# Constants
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 16000  # Sample rate
CHANNELS = 1  # Mono

# Initialize WhisperX model
model = whisperx.load_model("large-v2", device="cuda")  # Assuming you have a GPU for inference

# Function to process audio chunks
def process_audio_chunk(chunk):
    # Transcribe audio chunk
    result = model.transcribe(chunk)
    transcription = result["text"]
    print(transcription)  # Output transcription to console, you can modify this as needed

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Open stream
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK_SIZE)

print("Listening...")

# Main loop to capture and process audio
try:
    while True:
        # Read audio chunk from stream
        chunk = stream.read(CHUNK_SIZE)
        process_audio_chunk(chunk)
except KeyboardInterrupt:
    print("Stopped listening.")
    # Close the audio stream and PyAudio
    stream.stop_stream()
    stream.close()
    audio.terminate()
