import sounddevice as sd
import numpy as np
import whisper
import torch

model = whisper.load_model("medium")  # Choose the model

SAMPLE_RATE = 16000
CHUNK_DURATION = 0.75 # Change this value if necessary
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)

def callback(indata, frames, time, status):
    if status:
        print(status)
    audio_data = np.squeeze(indata)
    
    audio_tensor = torch.from_numpy(audio_data).float()
    
    result = model.transcribe(audio_tensor, language='en')
    transcribed_text = result['text'].strip()
    
    print(transcribed_text)
    
    if "exit program" in transcribed_text.lower():
        print("Exit command detected. Closing program.")
        raise sd.CallbackStop 

def start_transcription():
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback, blocksize=CHUNK_SIZE):
            print("Listening...")
            sd.sleep(int(1e10)) 
    except sd.CallbackStop:
        print("Program closed successfully.")

if __name__ == "__main__":
    start_transcription()
