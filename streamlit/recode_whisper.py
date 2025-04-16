# 필요한 라이브러리 임포트
import whisper      # OpenAI의 Whisper 음성 인식 모델
import pyaudio      # 오디오 입력 처리를 위한 라이브러리
import wave         # WAV 파일 처리를 위한 라이브러리
import os           # 파일 및 디렉토리 조작을 위한 라이브러리
import uuid         # 고유 식별자 생성을 위한 라이브러리
import warnings     # 경고 메시지 관리
from datetime import datetime  # 날짜와 시간 처리를 위한 라이브러리

# Whisper 모델 초기화
# 'base'는 가장 기본적인 모델이며, 'small', 'medium', 'large'로 변경하여 정확도를 높일 수 있음
model = whisper.load_model("large")  # "small"으로 바꾸면 정확도 상승
warnings.filterwarnings("ignore")    # 불필요한 경고 메시지 숨김

# 오디오 녹음 설정
FORMAT = pyaudio.paInt16    # 16비트 정수 포맷으로 오디오 녹음
CHANNELS = 1                # 모노 채널 사용
RATE = 16000               # 샘플링 레이트 16kHz (Whisper 권장 설정)
CHUNK = 1024               # 한 번에 처리할 오디오 청크 크기
RECORD_SECONDS = 6         # 각 녹음 세션의 길이 (초)
AUDIO_FOLDER = "data/recode/temp_audio" # 임시 오디오 파일이 저장될 폴더 경로

# 임시 오디오 파일을 저장할 폴더 생성
os.makedirs(AUDIO_FOLDER, exist_ok=True)

def record_audio(filename):
    """
    마이크로부터 오디오를 녹음하여 WAV 파일로 저장하는 함수
    
    Args:
        filename (str): 저장할 WAV 파일의 경로
    """
    # PyAudio 객체 초기화 및 스트림 설정
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                       rate=RATE, input=True,
                       frames_per_buffer=CHUNK)

    print("🎤 음성 입력 중... ({}초)".format(RECORD_SECONDS))
    frames = []  # 오디오 프레임을 저장할 리스트

    # 지정된 시간 동안 오디오 녹음
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("⏹ 녹음 종료")

    # 스트림 정리 및 종료
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # 녹음된 오디오를 WAV 파일로 저장
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

def realtime_whisper():
    """
    실시간으로 음성을 녹음하고 Whisper 모델을 사용하여 텍스트로 변환하는 함수
    Ctrl+C를 누르기 전까지 계속 실행됨
    """
    print("🚀 Whisper 실시간 음성 인식 시작 (종료하려면 Ctrl+C)")

    try:
        while True:
            # 고유한 파일명으로 임시 WAV 파일 생성
            filename = os.path.join(AUDIO_FOLDER, f"{uuid.uuid4()}.wav")
            
            # 오디오 녹음
            record_audio(filename)
            print("🧠 텍스트 변환 중...")

            # Whisper 모델을 사용하여 음성을 텍스트로 변환
            # language="ko"로 한국어 인식 지정
            result = model.transcribe(filename, language="ko")
            print("📝 인식된 텍스트:", result["text"])

            # 현재 날짜와 시간으로 파일명 생성
            current_time = datetime.now().strftime("%Y%m%d_%H")
            output_dir = "data/recode"
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{current_time}.txt")

            # 텍스트를 파일에 저장
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(f"{result['text']}\n")

            # 사용이 끝난 임시 파일 삭제
            os.remove(filename)

    except KeyboardInterrupt:
        print("\n🛑 프로그램을 종료합니다.")

# 스크립트가 직접 실행될 때만 realtime_whisper() 함수 실행
if __name__ == "__main__":
    realtime_whisper()