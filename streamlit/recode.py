import speech_recognition as sr
import pyaudio
import time

def realtime_speech_to_text():
    # 음성 인식기 초기화
    r = sr.Recognizer()
    
    # 마이크 사용 설정
    with sr.Microphone() as source:
        print("마이크 보정 중...")
        # 주변 소음 수준 조정
        r.adjust_for_ambient_noise(source)
        print("음성을 입력하세요 (종료하려면 Ctrl+C를 누르세요)")
        
        try:
            while True:
                print("\n듣는 중...")
                # 마이크로부터 음성 입력
                audio = r.listen(source)
                
                try:
                    # Google Speech Recognition을 사용하여 음성을 텍스트로 변환
                    text = r.recognize_google(audio, language='ko-KR')
                    print("인식된 텍스트:", text)
                    
                except sr.UnknownValueError:
                    print("음성을 인식할 수 없습니다")
                except sr.RequestError as e:
                    print("Google Speech Recognition 서비스 에러; {0}".format(e))
                    
                time.sleep(0.2)  # 잠시 대기
                
        except KeyboardInterrupt:
            print("\n프로그램을 종료합니다.")

if __name__ == "__main__":
    realtime_speech_to_text()