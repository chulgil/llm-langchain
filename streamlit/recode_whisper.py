# 필요한 라이브러리 임포트
import whisper      # OpenAI의 Whisper 음성 인식 모델
import pyaudio      # 오디오 입력 처리를 위한 라이브러리
import wave         # WAV 파일 처리를 위한 라이브러리
import os           # 파일 및 디렉토리 조작을 위한 라이브러리
import uuid         # 고유 식별자 생성을 위한 라이브러리
import warnings     # 경고 메시지 관리
from datetime import datetime  # 날짜와 시간 처리를 위한 라이브러리
import concurrent.futures  # 동시 실행을 위한 라이브러리
import logging        # 로깅을 위한 라이브러리
import torch         # PyTorch 라이브러리
import atexit        # 프로그램 종료 시 정리 작업을 위한 라이브러리
import gc            # 가비지 컬렉션을 위한 라이브러리
from dotenv import load_dotenv  # .env 파일 로드를 위한 라이브러리
import queue         # 오디오 버퍼 처리를 위한 큐

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# .env 파일 로드
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(env_path)

# 환경 변수에서 설정 로드
ENABLE_ENGLISH = os.getenv("ENABLE_ENGLISH_TRANSCRIPTION", "false").lower() == "true"

# Whisper 모델 초기화 - 성능 향상을 위해 small 모델 사용 (small, medium, large)
model = whisper.load_model("turbo", device="cpu")
warnings.filterwarnings("ignore")

# 오디오 설정
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 2048  # 최적의 청크 크기로 조정
BUFFER_SECONDS = 5  # 버퍼에 저장할 오디오 길이 (초)
BUFFER_SIZE = int(RATE * BUFFER_SECONDS / CHUNK) * CHUNK

# 폴더 경로 설정
BASE_DIR = "data/recode"
AUDIO_FOLDER = os.path.join(BASE_DIR, "temp_audio")
OUTPUT_FOLDER_KR = os.path.join(BASE_DIR, "kr")
OUTPUT_FOLDER_EN = os.path.join(BASE_DIR, "en")

def create_directories():
    """필요한 모든 디렉토리를 생성하는 함수"""
    directories = [AUDIO_FOLDER, OUTPUT_FOLDER_KR, OUTPUT_FOLDER_EN]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def record_audio(audio_queue):
    """오디오를 실시간으로 녹음하여 큐에 저장하는 함수"""
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                       rate=RATE, input=True,
                       frames_per_buffer=CHUNK)

    print("🎤 음성 입력 중... (종료하려면 Ctrl+C)")
    
    try:
        while True:
            data = stream.read(CHUNK)
            audio_queue.put(data)
    except KeyboardInterrupt:
        print("\n⏹ 녹음 종료")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

def process_audio_buffer(audio_queue, output_file_kr, output_file_en):
    """오디오 버퍼를 처리하여 텍스트로 변환하는 함수"""
    audio_buffer = []
    
    while True:
        try:
            # 오디오 데이터를 큐에서 가져와 버퍼에 추가
            data = audio_queue.get(timeout=1)
            audio_buffer.append(data)
            
            # 버퍼가 충분히 채워졌을 때 처리
            if len(audio_buffer) >= BUFFER_SIZE / CHUNK:
                # 오디오 데이터를 WAV 파일로 저장
                filename = os.path.join(AUDIO_FOLDER, f"{uuid.uuid4()}.wav")
                with wave.open(filename, 'wb') as wf:
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
                    wf.setframerate(RATE)
                    wf.writeframes(b''.join(audio_buffer))
                
                # 버퍼 초기화
                audio_buffer = []
                
                # 텍스트 변환
                print("🧠 텍스트 변환 중...")
                
                # 처리할 언어 목록 설정
                languages = ["ko"]
                if ENABLE_ENGLISH:
                    languages.append("en")
                
                # 각 언어별로 별도의 스레드에서 처리
                futures = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(languages)) as executor:
                    # 모든 언어에 대한 변환 작업을 동시에 시작
                    for lang in languages:
                        future = executor.submit(transcribe_audio, filename, lang)
                        futures.append((future, lang))
                    
                    # 모든 결과를 기다렸다가 한 번에 처리
                    for future, lang in futures:
                        try:
                            _, text = future.result()
                            if text.strip():  # 빈 텍스트가 아닌 경우에만 저장
                                output_file = output_file_kr if lang == "ko" else output_file_en
                                print(f"📝 인식된 {'한국어' if lang == 'ko' else '영어'}: {text}")
                                with open(output_file, "a", encoding="utf-8") as f:
                                    f.write(f"{text}\n")
                        except Exception as e:
                            logger.error(f"{lang} 변환 중 오류 발생: {str(e)}")
                
                # 임시 파일 삭제
                if os.path.exists(filename):
                    os.remove(filename)
                    
        except queue.Empty:
            continue
        except KeyboardInterrupt:
            break

def transcribe_audio(filename, language):
    """음성을 특정 언어로 변환하는 함수"""
    try:
        # 각 스레드마다 새로운 모델 인스턴스 생성
        model = whisper.load_model("small", device="cpu")
        # 처리 속도 향상을 위해 beam_size와 temperature 조정
        result = model.transcribe(
            filename, 
            language=language,
            beam_size=1,  # 빔 크기 감소로 처리 속도 향상
            temperature=0.0  # 결정적 결과를 위해 온도 0으로 설정
        )
        return language, result["text"]
    except Exception as e:
        logger.error(f"{language} 변환 중 오류: {str(e)}")
        raise

def realtime_whisper():
    """실시간으로 음성을 녹음하고 텍스트로 변환하는 함수"""
    print("🚀 Whisper 실시간 음성 인식 시작 (종료하려면 Ctrl+C)")
    print(f"영어 변환 {'활성화' if ENABLE_ENGLISH else '비활성화'}")
    print(f"처리할 언어: {['ko', 'en'] if ENABLE_ENGLISH else ['ko']}")

    # 필요한 디렉토리 생성
    create_directories()

    # 현재 날짜와 시간으로 파일명 생성
    current_time = datetime.now().strftime("%Y%m%d_%H")
    output_file_kr = os.path.join(OUTPUT_FOLDER_KR, f"{current_time}.txt")
    output_file_en = os.path.join(OUTPUT_FOLDER_EN, f"{current_time}.txt")
    
    # 파일이 없으면 생성
    for file_path in [output_file_kr, output_file_en]:
        if not os.path.exists(file_path):
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("")

    # 오디오 큐 생성
    audio_queue = queue.Queue()

    try:
        # 녹음과 처리를 별도의 스레드에서 실행
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            record_future = executor.submit(record_audio, audio_queue)
            process_future = executor.submit(process_audio_buffer, audio_queue, output_file_kr, output_file_en)
            
            # 두 작업이 완료될 때까지 대기
            concurrent.futures.wait([record_future, process_future])
            
    except KeyboardInterrupt:
        print("\n🛑 프로그램을 종료합니다.")
    except Exception as e:
        logger.error(f"예상치 못한 오류 발생: {str(e)}")
    finally:
        cleanup_resources()

def cleanup_resources():
    """리소스 정리 함수"""
    gc.collect()

if __name__ == "__main__":
    realtime_whisper()