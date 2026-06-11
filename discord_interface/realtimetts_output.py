from RealtimeTTS import TextToAudioStream, CoquiEngine
import logging
import stanza
import time
import uuid
import re
from g2pk import G2p

def generate_with_realtime_tts(text):
    timestamp = int(time.time() * 1000)
    unique_id = uuid.uuid4().hex
    file_name = f"output_temp/output_{timestamp}_{unique_id}.wav"
    logging.basicConfig(level=logging.INFO)
    engine = CoquiEngine(voice="female_korean", language="ko", level=logging.INFO)
    stream = TextToAudioStream(engine)

    text = re.sub(r'[^A-Za-z0-9가-힣\s]', '', text)

    phonemes = G2p()(text)

    stream.feed(phonemes)

    stanza.download(lang="ko",package=None,processors={"pos":"kaist_nocharlm"})
    stanza.download(lang="ko",package=None,processors={"lemma":"kaist_nocharlm"})
    stanza.download(lang="ko",package=None,processors={"depparse":"kaist_nocharlm"})
    tokenizer = "stanza"
    sentence_length = 2

    stream.play(output_wavfile=file_name, language="ko", tokenizer=tokenizer, minimum_sentence_length=sentence_length, minimum_first_fragment_length=sentence_length, context_size=sentence_length, debug=True)
    time.sleep(5)
    engine.shutdown()

if __name__ == '__main__':
    generate_with_realtime_tts("안녕하세요. 지금은 새로운 '티티에스'를 테스트하고 있습니다. 1 / 2 : 3 잘 들리시나요?")

# realtime-tts는 tts 짬뽕임. tts모델 이것 저것 다 넣어놓긴 했는데 그냥 직접 구현하는게 좋아보임.
# 그래도 이거 보면서 다양한 tts 모델 사용하는 방법을 익힐 수 있었음.