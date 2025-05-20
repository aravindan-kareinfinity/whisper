from gtts import gTTS

# Read the test text
with open('test.txt', 'r') as file:
    text = file.read()

# Create audio file
tts = gTTS(text=text, lang='en')
tts.save('test_audio.mp3')
print("Audio file 'test_audio.mp3' has been created!") 