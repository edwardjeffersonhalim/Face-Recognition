import sounddevice as sd
import numpy as np
import whisper
import torch
import threading

model = whisper.load_model("small")  # Choose the model

# Recipe
SAMPLE_RATE = 16000
CHUNK_DURATION = 1 
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
BUFFER_DURATION = 2  
OVERLAP = 0.5  

audio_buffer = []
transcription_thread = None
previous_text = ""

def transcribe_audio(buffer):
    global previous_text

    audio_tensor = torch.cat(buffer)
    result = model.transcribe(audio_tensor, language='en')
    transcribed_text = result['text'].strip()

    if transcribed_text.startswith(previous_text):
        transcribed_text = transcribed_text[len(previous_text):].strip()
    
    if transcribed_text:
        print(transcribed_text)
    
    previous_text = transcribed_text
    
    if "exit program" in transcribed_text.lower():
        print("Exit command detected. Closing program.")
        raise sd.CallbackStop 

def callback(indata, frames, time, status):
    global transcription_thread

    if status:
        print(status)

    audio_data = np.squeeze(indata)
    audio_tensor = torch.from_numpy(audio_data).float()

    audio_buffer.append(audio_tensor)

    if len(audio_buffer) * CHUNK_DURATION >= BUFFER_DURATION:
        if transcription_thread is None or not transcription_thread.is_alive():
            transcription_thread = threading.Thread(target=transcribe_audio, args=(audio_buffer,))
            transcription_thread.start()

        audio_buffer[:] = audio_buffer[-int(OVERLAP * SAMPLE_RATE):]

def start_transcription():
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback, blocksize=CHUNK_SIZE):
            print("Listening...")
            sd.sleep(int(1e10)) 
    except sd.CallbackStop:
        print("Program closed successfully.")

if __name__ == "__main__":
    start_transcription()
