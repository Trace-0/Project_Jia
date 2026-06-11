import pyttsx3

engine = pyttsx3.init()

def generate_with_pyttsx3(text):
    voices = engine.getProperty('voices')
    for i, voice in enumerate(voices):
        print(f"Voice {i}: {voice.name} - {voice.languages}")

    for voice in voices:
        if "korean" in voice.name.lower() or "ko_" in voice.id.lower():
            engine.setProperty('voice', voice.id)
            break
    
    engine.say(text)
    engine.runAndWait()

generate_with_pyttsx3("안녕하세요. 지금은 pyttsx3 테스트 중입니다. 잘 들리시나요?")

# 너무 구림