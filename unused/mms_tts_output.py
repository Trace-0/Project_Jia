from transformers import VitsModel, AutoTokenizer
import torch
import scipy.io.wavfile as wavfile

model = VitsModel.from_pretrained("facebook/mms-tts-kor")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-kor")

def generate_with_mms_tts_kor(text):
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        output = model(**inputs).waveform

        wavfile.write("techno.wav", rate=model.config.sampling_rate, data=output.numpy().squeeze())

generate_with_mms_tts_kor("안녕하세요. 저는 다국어 음성 합성 모델인 MMS-TTS-KOR입니다. 만나서 반가워요!")

# 웨째서 남자 보이스임? 여자 보이스로 바꿀 수 없음?
# 지아 여캔데 