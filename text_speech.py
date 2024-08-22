from gtts import gTTS
from playsound import playsound
import os

# Current available english voicings based on the documentation
voice_options = {
    '1': {'lang': 'en', 'tld': 'ca' ,'name': 'English (Default)', 'example_text': 'Hello, this is the default English voice.'},
    '2': {'lang': 'en-au', 'tld': 'com.au','name': 'English (Australian)', 'example_text': 'G’day mate, this is the Australian voice.'},
    '3': {'lang': 'en-uk', 'tld': 'co.uk', 'name': 'English (British)', 'example_text': 'Hello, this is the British English voice.'},
    '4': {'lang': 'en-us', 'tld': 'us', 'name': 'English (American)', 'example_text': 'Hi there, this is the American voice.'},
    '5': {'lang': 'en-in', 'tld': 'co.in', 'name': 'English (Indian)', 'example_text': 'Namaste, this is the Indian English voice.'}
}

# Function to output sample voices for user
def play_voice_sample(text, lang, slow):
    sample_file = "speed_sample.mp3"
    tts = gTTS(text=text, lang=lang, slow=slow)
    tts.save(sample_file)
    playsound(sample_file)
    os.remove(sample_file)

# Function to remove the sample files after they have been played
def clean_up():
    for file in os.listdir():
        if file.startswith("voice_sample_") and file.endswith(".mp3"):
            os.remove(file)
        if file.startswith("sentence") and file.endswith("mp3"):
            os.remove(file)

# Give an example of normal and slow speech
def get_user_preferences():
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

        while True:
            input_method = input("Would you like to (1) enter your prompt manually or (2) read from a text file? Enter 'q' to quit: ").lower()
            
            if input_method == '1':
                answer = input("Enter your prompt (Press 'q' to quit): ")
                if answer == "q":
                    print("Exiting the program...")
                    clean_up()
                    return
                else:
                    tts = gTTS(text=answer, lang=selected_voice, slow=slow)
                    tts.save("speech.mp3")
                    playsound("speech.mp3")
                    os.remove("speech.mp3")
            
            elif input_method == '2':
                while True:
                    file_path = input("Enter the path to the text file (or 'b' to go back): ")
                    if file_path.lower() == 'b':
                        break 
                    try:
                        with open(file_path, 'r') as file:
                            content = file.read()
                        tts = gTTS(text=content, lang=selected_voice, slow=slow)
                        tts.save("file_speech.mp3")
                        playsound("file_speech.mp3")
                        
                        while True:
                            replay_choice = input("\nWould you like to (1) replay the speech, (2) go back to the main menu, or (3) quit? Enter 1, 2, or 3: ").lower()
                            if replay_choice == "1":
                                playsound("file_speech.mp3")
                            elif replay_choice == "2":
                                break 
                            elif replay_choice == "3":
                                print("Exiting...")
                                os.remove("file_speech.mp3")
                                clean_up()
                                return
                            else:
                                print("Invalid input! Please choose '1', '2', or '3'.")
                        os.remove("file_speech.mp3")
                        break 
                    except FileNotFoundError:
                        print(f"Error: File not found at {file_path}. Please check the path and try again.")
                        continue

            elif input_method == 'q':
                print("Exiting the program...")
                clean_up()
                return
            else:
                print("Invalid input! Please choose '1' or '2' (or 'q' to quit).")
                continue

if __name__ == "__main__":
    main()
