import cv2
import numpy as np

# 讀取影片
cap = cv2.VideoCapture('transfer_video.mp4')  # 請確保 'input_video.mp4' 是您的影片檔案名稱

# 獲取影片的參數
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 設定輸出影片的參數，保存為 MP4 格式
out = cv2.VideoWriter('catch_coin_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# 定義 ROI 範圍 (左上角 x, y 以及寬度和高度)
roi_x, roi_y, roi_w, roi_h = 20, 110, 600, 400  # 範例：從 (100, 100) 開始，大小為 600x400

paused = False  # 控制播放/暫停的變數

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 複製原始幀
    output = frame.copy()

    
    # 取得 ROI 區域
    roi = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 直方圖均衡化
    gray = cv2.equalizeHist(gray)

    # 高斯模糊去噪
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # 使用CLAHE進行對比度增強
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(blurred)


    # 使用霍夫變換來檢測圓形
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=50,
        param1=50,
        param2=30,
        minRadius=10,
        maxRadius=25
    )

    # 繪製檢測到的圓形並標記代號
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for idx, i in enumerate(circles[0, :]):
            # 畫圓形輪廓
            cv2.circle(output, (i[0] + roi_x, i[1] + roi_y), i[2], (0, 255, 0), 2)
            # 標記圓心
            cv2.circle(output, (i[0] + roi_x, i[1] + roi_y), 2, (0, 0, 255), 3)
            # 標記代號 (C1, C2, C3...)
            label = f"C{idx + 1}"
            cv2.putText(output, label, (i[0] + roi_x - 20, i[1] + roi_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # 繪製ROI範圍的矩形框
    cv2.rectangle(output, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)

    # 寫入處理後的幀到輸出影片
    out.write(output)

    # 顯示結果
    cv2.imshow('Detected Coins with Labels', output)
    cv2.imshow('gray', gray)
    # cv2.imshow('edges', edges)

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
