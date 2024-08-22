import sounddevice as sd
import numpy as np
import speech_recognition as sr
from gtts import gTTS
import os
import tempfile
from datetime import datetime
import time
from textblob import TextBlob

def speak_text(text):
    # Initialize
    tts = gTTS(text)
    
    # Save the output as a temporary file
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as fp:
        tts.save(fp.name)
        temp_path = fp.name
    
    # Play the generated speech
    os.system(f"afplay {temp_path}" if os.name == "posix" else f"start {temp_path}")
    
    # Clean up the temporary file
    os.remove(temp_path)

def analyze_sentiment(text):
    # Analyze sentiment using TextBlob
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    
    # Sentiment Output
    if sentiment_score > 0:
        sentiment = "Positive"
    elif sentiment_score < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    
    return sentiment, sentiment_score

def record_text(play_prompt=False, duration=8):
    sample_rate = 16000

    try:
        if play_prompt:
            # Inform the user to start the speech
            speak_text("You may speak now")
            time.sleep(1)
        
        print("Recording...")

        # Recording with sounddevice
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
        sd.wait()

        # Convert file so it is ready to be read
        audio_bytes = np.array(audio).tobytes()

        # Use speech_recognition to transcribe
        r = sr.Recognizer()
        audio_data = sr.AudioData(audio_bytes, sample_rate, 2)
        text = r.recognize_google(audio_data)
        text = text.lower()

        return text

    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))
    except sr.UnknownValueError:
        print("Unknown error occurred")
    return ""

def output_text(text, sentiment, sentiment_score):
    # Get the current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_line = f"[{timestamp}] {text} | Sentiment: {sentiment} (Score: {sentiment_score})\n"
    
    # Write the output line to the text file
    with open("output.txt", "a") as f:
        f.write(output_line)

    # Print to the terminal as well
    print(output_line)

if __name__ == "__main__":
    text = record_text(play_prompt=True, duration=8)

    if text:
        sentiment, sentiment_score = analyze_sentiment(text)
        output_text(text, sentiment, sentiment_score)

    while True:
        text = record_text(duration=8)
        if text:
            print(f"Recognized: {text}")

            # Exit the program if "exit program" is said
            if "exit program" in text:
                print("Exiting program...")
                break

            sentiment, sentiment_score = analyze_sentiment(text)
            output_text(text, sentiment, sentiment_score)

'''
Positive if the score is greater than 0.
Negative if the score is less than 0.
Neutral if the score is exactly 0.
'''