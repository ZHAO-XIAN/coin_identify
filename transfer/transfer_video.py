import cv2
import numpy as np

# 讀取影片
video_path = '../video/755776412.816110.mp4'  # 確保這裡是您的影片檔案路徑
cap = cv2.VideoCapture(video_path)

# 檢查影片是否成功讀取
if not cap.isOpened():
    print("Error: 無法讀取影片。")
    exit()

# 定義源點 (您提供的四個點)
src_points = np.float32([
    (85, 807),
    (448, 655),
    (681, 770),
    (234, 1050)
])

# 定義目標點 (將源點映射到的矩形區域)
dst_points = np.float32([
    (0, 0),
    (720, 0),
    (720, 480),
    (0, 480)
])

# 計算透視變換矩陣
M = cv2.getPerspectiveTransform(src_points, dst_points)

# 取得影片的幀率和解析度，用於保存結果
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = 720
frame_height = 480

# 設定影片輸出格式 (使用 MJPG 格式編碼)
out = cv2.VideoWriter('transfer_video_test.mp4', cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))
paused = False

# 逐幀處理影片
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # 影片結束

    # 進行透視變換
    warped_frame = cv2.warpPerspective(frame, M, (frame_width, frame_height), flags=cv2.INTER_CUBIC)

    # 顯示處理後的結果
    cv2.imshow('Warped Frame', warped_frame)

    # 保存處理後的幀到輸出影片
    out.write(warped_frame)

    # 按 'q' 鍵退出迴圈
    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):  # 按 'q' 退出
        break
    elif key == ord('p'):  # 按 'p' 暫停/繼續
        paused = not paused
        while paused:
            key_pause = cv2.waitKey(10) & 0xFF
            if key_pause == ord('p'):  # 按 'p' 恢復播放
                paused = False

# 釋放資源
cap.release()
out.release()
cv2.destroyAllWindows()
