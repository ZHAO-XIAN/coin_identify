import cv2
import numpy as np

# 讀取影片
cap = cv2.VideoCapture('transfer_video.mp4')  # 請確保 'input_video.mp4' 是您的影片檔案名稱

# 獲取影片的參數
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 設定輸出影片的參數，保存為 MP4 格式
out = cv2.VideoWriter('catch_coin_video_v2.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# 定義 ROI 範圍 (左上角 x, y 以及寬度和高度)
roi_x, roi_y, roi_w, roi_h = 20, 110, 600, 400  # 範例：從 (100, 100) 開始，大小為 600x400

paused = False  # 控制播放/暫停的變數

# 用來追蹤硬幣的ID和其位置
tracked_coins = {}  # key: 圓心位置，value: 代號

# 用來追蹤硬幣的位置變化
def find_matching_coin(new_center, threshold=40):
    """搜尋是否有與新檢測到的圓心相近的硬幣"""
    for key, (prev_center, label) in tracked_coins.items():
        if np.linalg.norm(np.array(new_center) - np.array(prev_center)) < threshold:
            return key, label  # 找到匹配的圓形，返回其ID
    return None, None  # 沒有找到匹配的圓形

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 複製原始幀
    output = frame.copy()

    # 取得 ROI 區域
    roi = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

    # 轉換為灰階圖像
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    gray = cv2.equalizeHist(gray)
    # 高斯模糊來去除噪點
    gray_blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # 使用霍夫變換來檢測圓形
    circles = cv2.HoughCircles(
        gray_blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=50,
        param1=50,
        param2=30,
        minRadius=10,
        maxRadius=25
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for idx, i in enumerate(circles[0, :]):
            circle_center = (i[0] + roi_x, i[1] + roi_y)
            radius = i[2]

            # 尋找是否有匹配的硬幣
            matching_key, label = find_matching_coin(circle_center)
            if matching_key is not None:
                # 如果找到了匹配的硬幣，更新位置並保持原來的代號
                tracked_coins[matching_key] = (circle_center, label)
            else:
                # 如果是新硬幣，給它一個新的代號
                label = f"C{len(tracked_coins) + 1}"
                tracked_coins[circle_center] = (circle_center, label)

            # 畫圓形輪廓
            cv2.circle(output, circle_center, radius, (0, 255, 0), 2)
            # 標記圓心
            cv2.circle(output, circle_center, 2, (0, 0, 255), 3)
            # 標記代號 (C1, C2, C3...)
            cv2.putText(output, label, (i[0] + roi_x - 20, i[1] + roi_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # 繪製ROI範圍的矩形框
    cv2.rectangle(output, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)

    # 寫入處理後的幀到輸出影片
    out.write(output)

    # 顯示結果
    cv2.imshow('Detected Coins with Labels', output)

    # 檢測按鍵，按 'q' 退出，按 'p' 暫停/繼續
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # 按 'q' 退出
        break
    elif key == ord('p'):  # 按 'p' 暫停/繼續
        paused = not paused
        while paused:
            key_pause = cv2.waitKey(1) & 0xFF
            if key_pause == ord('p'):  # 按 'p' 恢復播放
                paused = False

# 釋放資源
cap.release()
out.release()
cv2.destroyAllWindows()
