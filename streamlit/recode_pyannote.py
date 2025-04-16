# 필요한 라이브러리 임포트
import whisper
import pyaudio
import wave
import os
import uuid
import warnings
from datetime import datetime
from pyannote.audio import Pipeline
import torch
import numpy as np
from pydub import AudioSegment
import logging
from dotenv import load_dotenv
from pathlib import Path

# .env 파일 경로 설정
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# HuggingFace 토큰 설정
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    logger.error(f"HF_TOKEN이 .env 파일에 설정되지 않았습니다. (.env 파일 경로: {env_path})")
    raise ValueError("HF_TOKEN이 .env 파일에 설정되지 않았습니다.")

# Whisper 모델 초기화
model = whisper.load_model("large")
warnings.filterwarnings("ignore")

# Pyannote 파이프라인 초기화
try:
    logger.info("Pyannote 파이프라인 초기화 중...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HF_TOKEN,
        cache_dir=os.path.expanduser("~/.cache/pyannote")
    )
    logger.info("Pyannote 파이프라인 초기화 성공")
except Exception as e:
    logger.error(f"Pyannote 파이프라인 초기화 실패: {str(e)}")
    logger.error("다음 링크에서 모델 접근 권한을 요청해주세요:")
    logger.error("1. https://huggingface.co/pyannote/segmentation-3.0")
    logger.error("2. https://huggingface.co/pyannote/speaker-diarization-3.1")
    pipeline = None

# 오디오 녹음 설정
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 5
AUDIO_FOLDER = "data/recode/temp_audio" # 임시 오디오 파일이 저장될 폴더 경로
OUTPUT_FOLDER = "data/recode"

# 폴더 생성
os.makedirs(AUDIO_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def record_audio(filename):
    """마이크로부터 오디오를 녹음하여 WAV 파일로 저장하는 함수"""
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                       rate=RATE, input=True,
                       frames_per_buffer=CHUNK)

    print("🎤 음성 입력 중... ({}초)".format(RECORD_SECONDS))
    frames = []

    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("⏹ 녹음 종료")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

def extract_speaker_audio(input_file, start_time, end_time, output_file):
    """특정 시간 범위의 오디오를 추출하는 함수"""
    audio = AudioSegment.from_wav(input_file)
    segment = audio[start_time*1000:end_time*1000]  # 밀리초 단위로 변환
    segment.export(output_file, format="wav")

def realtime_diarization():
    """실시간으로 음성을 녹음하고 화자 분리 및 텍스트 변환을 수행하는 함수"""
    if pipeline is None:
        logger.error("Pyannote 파이프라인이 초기화되지 않았습니다. 프로그램을 종료합니다.")
        return

    print("🚀 실시간 화자 분리 및 텍스트 변환 시작 (종료하려면 Ctrl+C)")

    try:
        while True:
            # 임시 WAV 파일 생성
            temp_file = os.path.join(AUDIO_FOLDER, f"{uuid.uuid4()}.wav")
            record_audio(temp_file)

            print("🧠 화자 분리 중...")
            try:
                # 화자 분리 수행
                diarization = pipeline(temp_file)
                
                # 결과를 저장할 텍스트
                result_text = []
                
                # 각 화자별로 오디오 추출 및 텍스트 변환
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    start_time = turn.start
                    end_time = turn.end
                    
                    # 화자별 오디오 추출
                    speaker_audio = os.path.join(AUDIO_FOLDER, f"{uuid.uuid4()}_{speaker}.wav")
                    extract_speaker_audio(temp_file, start_time, end_time, speaker_audio)
                    
                    # Whisper로 텍스트 변환
                    result = model.transcribe(speaker_audio, language="ko")
                    result_text.append(f"{speaker}: {result['text']}")
                    
                    # 임시 파일 삭제
                    os.remove(speaker_audio)

                # 결과를 파일에 저장
                current_time = datetime.now().strftime("%Y%m%d_%H")
                output_file = os.path.join(OUTPUT_FOLDER, f"{current_time}.txt")
                
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write("\n".join(result_text) + "\n\n")
                
                print("📝 인식된 텍스트:")
                for line in result_text:
                    print(line)
                
            except Exception as e:
                logger.error(f"화자 분리 중 오류 발생: {str(e)}")
                print("화자 분리 중 오류가 발생했습니다. 다음 녹음으로 넘어갑니다.")
            
            finally:
                # 임시 파일 삭제
                if os.path.exists(temp_file):
                    os.remove(temp_file)

    except KeyboardInterrupt:
        print("\n🛑 프로그램을 종료합니다.")

if __name__ == "__main__":
    realtime_diarization()
