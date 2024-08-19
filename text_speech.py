from gtts import gTTS
from playsound import playsound
import os

# Available voicings
voice_options = {
    '1': {'lang': 'en', 'name': 'English (Default)', 'example_text': 'Hello, this is the default English voice.'},
    '2': {'lang': 'en-au', 'name': 'English (Australian)', 'example_text': 'G’day mate, this is the Australian voice.'},
    '3': {'lang': 'en-uk', 'name': 'English (British)', 'example_text': 'Hello, this is the British English voice.'},
    '4': {'lang': 'en-us', 'name': 'English (American)', 'example_text': 'Hi there, this is the American voice.'},
    '5': {'lang': 'en-in', 'name': 'English (Indian)', 'example_text': 'Namaste, this is the Indian English voice.'}
}

def play_voice_sample(text, lang, slow):
    # Generate a sample voice preview if it hasn't been done already
    sample_file = "speed_sample.mp3"
    tts = gTTS(text=text, lang=lang, slow=slow)
    tts.save(sample_file)
    playsound(sample_file)
    os.remove(sample_file)

def clean_up():
    # Remove the sample files after they have been played
    for file in os.listdir():
        if file.startswith("voice_sample_") and file.endswith(".mp3"):
            os.remove(file)

def get_user_preferences():
    # Give an example of normal and slow speech
    print("Here is an example of normal speech:")
    play_voice_sample("This is an example of normal speech speed.", 'en', slow=False)
    
    print("\nHere is an example of slow speech:")
    play_voice_sample("This is an example of slow speech speed.", 'en', slow=True)
    
    # Get speed preference
    while True:
        speed_choice = input("\nWould you like the speech to be slow (yes/no)? Default is 'no': ").lower()
        if speed_choice in ['yes', 'no', '']:
            slow = speed_choice == 'yes'
            break
        print("Invalid input. Please type 'yes' or 'no'.")
    
    # Get volume preference
    while True:
        try:
            volume = float(input("Set the volume (0.0 to 1.0). Default is 1.0: ") or "1.0")
            if 0.0 <= volume <= 1.0:
                break
            else:
                print("Volume must be between 0.0 and 1.0.")
        except ValueError:
            print("Invalid input. Please enter a number between 0.0 and 1.0.")
    
    return slow, volume

def main():
    print("Available voice options:\n")
    slow, volume = get_user_preferences()

    for key, voice in voice_options.items():
        print(f"{key}: {voice['name']}")
        play_voice_sample(voice['example_text'], voice['lang'], slow)
        print(f"Sample: {voice['example_text']}\n")

    while True:
        choice = input("Enter the number of the voice you want to use or 'q' to quit: ")

        if choice == 'q':
            print("Exiting...")
            clean_up()
            return

        if choice not in voice_options:
            print("Invalid choice! Please select a valid option.\n")
            continue

        selected_voice = voice_options[choice]['lang']
        print(f"You selected: {voice_options[choice]['name']}\n")

        # Main loop for text-to-speech conversion
        while True:
            answer = input("Enter your prompt (Press 'q' to quit): ")
            if answer == "q":
                print("Exiting the program...")
                clean_up()
                return

            tts = gTTS(text=answer, lang=selected_voice, slow=slow)
            tts.save("speech.mp3")
            playsound("speech.mp3")
            os.remove("speech.mp3")

if __name__ == "__main__":
    main()
