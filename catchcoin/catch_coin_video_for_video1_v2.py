import cv2
import numpy as np

"""
watershed
"""

# 讀取影片
cap = cv2.VideoCapture('../video/transfer_video.mp4')  # 請確保 'input_video.mp4' 是您的影片檔案名稱

# 獲取影片的參數
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 設定輸出影片的參數，保存為 MP4 格式
out = cv2.VideoWriter('catch_coin_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# 定義 ROI 範圍 (左上角 x, y 以及寬度和高度)
roi_x, roi_y, roi_w, roi_h = 50, 115, 500, 300  # 範例：從 (100, 100) 開始，大小為 600x400

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

# 定義白色面積的閾值 (調整該值以篩選目標區域)
AREA_THRESHOLD_MIN= 2200
AREA_THRESHOLD_MAX= 5000

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 複製原始幀
    output = frame.copy()

    # 取得 ROI 區域
    roi = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

    # Sobel 運算進行邊緣檢測
    normalized_magnitude = cv2.normalize(roi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 提升對比度並處理影像
    contrast = 200
    brightness = -100
    precess_img = normalized_magnitude * (contrast / 127 + 1) - contrast + brightness
    precess_img = np.clip(precess_img, 0, 255).astype(np.uint8)
    precess_img = cv2.GaussianBlur(precess_img, (11, 11), 0)
    
    precess_img = cv2.cvtColor(precess_img, cv2.COLOR_BGR2GRAY)

    # 邊緣檢測
    canny = cv2.Canny(precess_img, 38, 38)
    
    # 形態學操作 (膨脹)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    
    sure_fg = cv2.dilate(canny, kernel)

    # # 進行 watershed 分割
    # # 構造背景和前景的標籤圖像
    # dist_transform = cv2.distanceTransform(pre, cv2.DIST_L2, 5)
    # _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    
    sure_bg = cv2.dilate(sure_fg, kernel, iterations=3)
    sure_fg = np.uint8(sure_fg)
    
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 標註連通區域
    _, markers = cv2.connectedComponents(sure_fg)

    print('markers', markers)
    # 應用 watershed 算法
    markers = markers + 1
    markers[unknown == 255] = 0
    cv2.watershed(roi, markers)
    roi[markers == -1] = [0, 0, 255]

    # # 找出輪廓
    # _, contours, _ = cv2.findContours(pre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # # 遍歷所有輪廓
    # for contour in contours:
    #     # 計算輪廓的面積
    #     area = cv2.contourArea(contour)
    #     if AREA_THRESHOLD_MAX > area > AREA_THRESHOLD_MIN:  # 篩選面積大於閾值的區域
    #         # 計算外接矩形
    #         x, y, w, h = cv2.boundingRect(contour)
    #         # 在原圖上畫矩形框
    #         cv2.rectangle(output, (roi_x + x, roi_y + y), (roi_x + x + w, roi_y + y + h), (0, 255, 0), 2)
    #         # 標註面積
    #         cv2.putText(output, f"Area: {int(area)}", (roi_x + x, roi_y + y - 10),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # # 繪製 ROI 範圍的矩形框
    # cv2.rectangle(output, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)

    # 顯示結果
    cv2.imshow('Detected White Areas', output)
    cv2.imshow('roi', roi)
    # cv2.imshow('Processed Image', pre)
    cv2.imshow('sure_fg', sure_fg)
    cv2.imshow("sure_bg", sure_bg)

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
