# -*- coding: utf-8 -*-
"""DAVE(E2EE) 호환 디스코드 음성 수신 모듈

discord.py 2.7+에 내장된 DAVE 세션(davey)을 그대로 활용해서
discord-ext-voice-recv 없이 음성 패킷을 수신/복호화해요.
(discord-ext-voice-recv는 DAVE 미지원이라 E2EE 강제 적용 이후 수신이 불가능해요)

수신 경로:
  UDP 패킷 -> RTP 파싱 -> 전송 계층 복호화(aead_xchacha20_poly1305_rtpsize)
  -> DAVE(E2EE) 복호화 -> Opus 디코딩 -> AudioSink.write(member, VoiceData)
"""

from __future__ import annotations

import logging
import queue
import struct
import threading
import time
from typing import Optional

import davey
import discord
import nacl.secret
from discord.gateway import DiscordVoiceWebSocket
from discord.opus import Decoder
from discord.voice_state import VoiceConnectionState

logger = logging.getLogger(__name__)

OPUS_SILENCE = b'\xf8\xff\xfe'

# 수신 전용 게이트웨이 opcode (discord.py에는 수신 처리가 없는 것들)
_OP_VIDEO = 12  # user_id와 audio_ssrc 매핑이 들어있음


class VoiceData:
    """싱크로 전달되는 음성 데이터 (pcm: 48kHz / 16-bit / 2채널)"""

    __slots__ = ('pcm', 'opus', 'ssrc')

    def __init__(self, pcm: bytes, opus: bytes, ssrc: int):
        self.pcm = pcm
        self.opus = opus
        self.ssrc = ssrc


class AudioSink:
    """음성 수신 싱크 베이스 클래스 (voice_recv.AudioSink 대체)"""

    def __init__(self):
        self.voice_client: Optional[VoiceRecvClient] = None

    def wants_opus(self) -> bool:
        """True면 Opus 패킷 그대로, False면 디코딩된 PCM을 받아요"""
        return False

    def write(self, user: Optional[discord.Member], data: VoiceData):
        raise NotImplementedError

    def cleanup(self):
        pass


class AudioReceiver(threading.Thread):
    """소켓 콜백으로 받은 패킷을 복호화/디코딩해서 싱크로 전달하는 워커 스레드"""

    KEEPALIVE_INTERVAL = 5.0  # NAT 매핑 유지를 위한 UDP keepalive 주기(초)

    def __init__(self, sink: AudioSink, client: VoiceRecvClient):
        super().__init__(daemon=True, name=f'voice-receiver-{id(self):x}')
        self.sink = sink
        self.client = client
        self._queue: queue.SimpleQueue[Optional[bytes]] = queue.SimpleQueue()
        self._decoders: dict[int, Decoder] = {}
        self._box: Optional[nacl.secret.Aead] = None
        self._end = threading.Event()
        self._keepalive = threading.Thread(
            target=self._keepalive_loop, daemon=True, name=f'voice-udp-keepalive-{id(self):x}'
        )

        mode = client._connection.mode
        if mode != 'aead_xchacha20_poly1305_rtpsize':
            raise RuntimeError(f'지원하지 않는 음성 암호화 모드예요: {mode}')
        self.update_secret_key(bytes(client._connection.secret_key))

    # ==== 수명 주기 ====

    def start(self):
        self.client._connection.add_socket_listener(self.feed)
        self._keepalive.start()
        super().start()

    def stop(self):
        if self._end.is_set():
            return
        self._end.set()
        self.client._connection.remove_socket_listener(self.feed)
        self._queue.put(None)  # run 루프 깨우기

    def update_secret_key(self, secret_key: bytes):
        self._box = nacl.secret.Aead(secret_key)

    def destroy_decoder(self, ssrc: int):
        self._decoders.pop(ssrc, None)

    # ==== 패킷 처리 ====

    def feed(self, data: bytes):
        """SocketReader 스레드에서 호출되는 콜백. 무거운 처리는 워커로 넘겨요"""
        if self._end.is_set() or len(data) < 28:
            return
        if data[0] >> 6 != 2:  # RTP 버전 비트 확인 (IP discovery/keepalive 응답 제외)
            return
        if 200 <= data[1] <= 204:  # RTCP는 사용하지 않음
            return
        self._queue.put(data)

    def run(self):
        while not self._end.is_set():
            data = self._queue.get()
            if data is None:
                continue
            try:
                self._process(data)
            except Exception:
                logger.exception('음성 패킷 처리 중 오류가 발생했어요')
        try:
            self.sink.cleanup()
        except Exception:
            logger.exception('싱크 cleanup 중 오류가 발생했어요')

    def _process(self, data: bytes):
        decrypted = self._decrypt_rtp(data)
        if decrypted is None:
            return
        ssrc, payload = decrypted

        if payload == OPUS_SILENCE:
            return

        user_id = self.client._ssrc_to_id.get(ssrc)
        if user_id is None:
            # SPEAKING/VIDEO 이벤트가 오기 전까지는 발신자를 알 수 없으니 버려요
            return

        # DAVE(E2EE) 복호화 - discord.py가 유지하는 세션을 그대로 사용
        state = self.client._connection
        if state.dave_protocol_version > 0 and state.dave_session is not None:
            try:
                opus_frame = state.dave_session.decrypt(user_id, davey.MediaType.audio, bytes(payload))
            except Exception as e:
                logger.debug('DAVE 복호화 실패 (ssrc=%s, user=%s): %s', ssrc, user_id, e)
                return
        else:
            opus_frame = bytes(payload)

        if self.sink.wants_opus():
            pcm = b''
        else:
            decoder = self._decoders.get(ssrc)
            if decoder is None:
                decoder = self._decoders[ssrc] = Decoder()
            try:
                pcm = decoder.decode(opus_frame, fec=False)
            except Exception as e:
                logger.debug('Opus 디코딩 실패 (ssrc=%s): %s', ssrc, e)
                return

        member = self.client.guild.get_member(user_id)
        self.sink.write(member, VoiceData(pcm=pcm, opus=opus_frame, ssrc=ssrc))

    def _decrypt_rtp(self, data: bytes) -> Optional[tuple[int, bytes]]:
        """aead_xchacha20_poly1305_rtpsize 전송 계층 복호화. (ssrc, payload) 반환"""
        first = data[0]
        cc = first & 0x0F
        extended = bool(first & 0x10)
        ssrc = struct.unpack_from('>I', data, 8)[0]

        header = data[:12]
        body = data[12 + cc * 4:]

        nonce = bytearray(24)
        nonce[:4] = data[-4:]

        # rtpsize 모드: RTP 헤더(+확장 헤더 첫 4바이트)는 평문 AAD, 나머지가 암호문
        if extended:
            aad = bytes(header) + bytes(body[:4])
            ciphertext = body[4:-4]
        else:
            aad = bytes(header)
            ciphertext = body[:-4]

        try:
            plaintext = self._box.decrypt(bytes(ciphertext), aad, bytes(nonce))
        except Exception as e:
            logger.debug('전송 계층 복호화 실패 (ssrc=%s): %s', ssrc, e)
            return None

        if extended:
            # 확장 헤더 데이터(length * 4바이트)는 복호화된 페이로드 앞부분에 있음
            ext_length = struct.unpack_from('>H', aad, 14)[0]
            plaintext = plaintext[ext_length * 4:]

        return ssrc, plaintext

    # ==== UDP keepalive ====

    def _keepalive_loop(self):
        """수신만 할 때도 NAT 매핑이 닫히지 않게 주기적으로 UDP 패킷을 보내요"""
        counter = 0
        while not self._end.wait(self.KEEPALIVE_INTERVAL):
            state = self.client._connection
            if not state.is_connected():
                continue
            try:
                packet = counter.to_bytes(8, 'big')
                state.socket.sendto(packet, (state.endpoint_ip, state.voice_port))
                counter = (counter + 1) % (1 << 64)
            except Exception as e:
                logger.debug('UDP keepalive 전송 실패: %s', e)


class VoiceRecvClient(discord.VoiceClient):
    """DAVE 호환 음성 수신을 지원하는 VoiceClient

    사용법: ``await voice_channel.connect(cls=VoiceRecvClient)`` 후 ``listen(sink)``
    """

    def __init__(self, client: discord.Client, channel: discord.abc.Connectable):
        super().__init__(client, channel)
        self._ssrc_to_id: dict[int, int] = {}
        self._receiver: Optional[AudioReceiver] = None

    def create_connection_state(self) -> VoiceConnectionState:
        # 게이트웨이 훅을 등록해서 SPEAKING 등 수신 이벤트로 ssrc-유저 매핑을 유지
        return VoiceConnectionState(self, hook=self._ws_hook)

    async def _ws_hook(self, ws: DiscordVoiceWebSocket, msg: dict):
        op = msg.get('op')
        data = msg.get('d') or {}

        if op == DiscordVoiceWebSocket.SPEAKING:
            # 유저가 말하기 시작할 때 ssrc가 내려와요
            self._ssrc_to_id[data['ssrc']] = int(data['user_id'])

        elif op == _OP_VIDEO:
            if data.get('audio_ssrc'):
                self._ssrc_to_id[data['audio_ssrc']] = int(data['user_id'])

        elif op == DiscordVoiceWebSocket.CLIENT_DISCONNECT:
            user_id = int(data['user_id'])
            for ssrc in [s for s, uid in self._ssrc_to_id.items() if uid == user_id]:
                del self._ssrc_to_id[ssrc]
                if self._receiver:
                    self._receiver.destroy_decoder(ssrc)

        elif op == DiscordVoiceWebSocket.SESSION_DESCRIPTION:
            # 재연결 등으로 secret key가 바뀌면 갱신
            if self._receiver and ws.secret_key:
                self._receiver.update_secret_key(bytes(ws.secret_key))

    def listen(self, sink: AudioSink):
        """싱크를 등록하고 음성 수신을 시작해요"""
        if not self.is_connected():
            raise discord.ClientException('음성 채널에 연결된 상태가 아니에요')
        if self._receiver is not None:
            raise discord.ClientException('이미 음성을 수신하고 있어요')

        sink.voice_client = self
        self._receiver = AudioReceiver(sink, self)
        self._receiver.start()
        logger.info('음성 수신을 시작했어요 (DAVE 프로토콜 v%d)', self._connection.dave_protocol_version)

    def stop_playback(self):
        """재생 중인 오디오만 멈춰요 (음성 수신은 유지)"""
        super().stop()

    def stop_listening(self):
        """음성 수신을 중단해요"""
        if self._receiver is not None:
            self._receiver.stop()
            self._receiver = None

    def is_listening(self) -> bool:
        return self._receiver is not None

    def stop(self):
        super().stop()
        self.stop_listening()
