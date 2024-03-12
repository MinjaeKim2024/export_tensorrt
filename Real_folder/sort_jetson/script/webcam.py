import cv2
import os

def generate_filename(base_name="output", ext=".mp4"):
    """
    기존에 존재하지 않는 파일 이름을 생성하여 반환합니다.
    """
    counter = 1
    new_filename = f"{base_name}{ext}"
    while os.path.exists(new_filename):
        new_filename = f"{base_name}{counter}{ext}"
        counter += 1
    return new_filename

# 새로운 파일 이름 생성
filename = generate_filename()

# 웹캠 설정
cap = cv2.VideoCapture(0)

# 비디오 저장을 위한 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 코덱 설정
out = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        # 비디오 저장
        out.write(frame)

        # 화면에 프레임 표시
        cv2.imshow('frame', frame)

        # 'q' 키를 누르면 루프 탈출
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# 작업 완료 후 해제
cap.release()
out.release()
cv2.destroyAllWindows()
