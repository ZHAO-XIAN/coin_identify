import cv2
import numpy as np
import json
import statistics
from scipy.stats import norm

"""
修改分辨頂部及底部
修改前後景分離
"""

class CoinTracker:
    def __init__(self, roi):
        self.roi = roi
        self.min_radius = 10
        self.max_radius = 25
        self.tracked_coins = {}  # 用於追蹤硬幣的ID和其位置
        self.coin_records = {}  # 用於記錄硬幣的詳細信息
        self.frame_counter = 0  # 當前幀計數
        self.threshold = 40  # 用於確定硬幣是否匹配的閾值

    def find_matching_coin(self, new_center):
        """搜尋是否有與新檢測到的圓心相近的硬幣"""
        for key, (prev_center, label) in self.tracked_coins.items():
            if np.linalg.norm(np.array(new_center) - np.array(prev_center)) < self.threshold:
                return key, label  # 找到匹配的圓形，返回其ID
        return None, None  # 沒有找到匹配的圓形

    def format_time(self, milliseconds, fps):
        """將幀數轉換為時間格式 (MM:SS:MMM)"""
        seconds = milliseconds // fps
        milliseconds = int((milliseconds % fps) * (1000 / fps))
        return f"{seconds // 60}:{seconds % 60}:{milliseconds}"

    def process_frame(self, frame):
        roi_x, roi_y, roi_w, roi_h = self.roi
        roi = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

        # 轉換為灰階圖像
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        gray_blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        prep_frame = 255 - gray_blurred

        # 更新當前幀計數
        self.frame_counter += 1
        return prep_frame

    def find_circles(self, prep_frame):
        # 使用霍夫變換來檢測圓形
        circles = cv2.HoughCircles(
            prep_frame,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=50,
            param1=50,
            param2=30,
            minRadius=self.min_radius,
            maxRadius=self.max_radius
        )
        return circles

    def coin_top_or_bottom(self, record):
        """
        判斷硬幣是在頂部還是底部，抓取最大及最小的Y座標，計算中心點的Y座標
        """
        if len(record["circle_center"]) > 2:
            max_circle_center_y = max(record["circle_center"], key=lambda x: x[1])[1]
            min_circle_center_y = min(record["circle_center"], key=lambda x: x[1])[1]
            distance = (max_circle_center_y - min_circle_center_y)/2
            if record["circle_center"][-1][1] > min_circle_center_y + distance: 
                return "bottom"
            else:
                return "top"
        else:
            return "top"

    def coin_tracking(self, frame, circles , fps):
        frame = frame.copy()
        roi_x, roi_y, roi_w, roi_h = self.roi
        position = "top"

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for idx, i in enumerate(circles[0, :]):
                circle_center = (i[0] + roi_x, i[1] + roi_y)
                radius = i[2]

                # 尋找是否有匹配的硬幣
                matching_key, label = self.find_matching_coin(circle_center)

                if matching_key is not None:
                    # 如果找到了匹配的硬幣，更新位置並保持原來的代號
                    self.tracked_coins[matching_key] = (circle_center, label)
                    record = self.coin_records[label]
                    record["disappear_time"] = self.format_time(self.frame_counter, fps)
                    record["circle_center"].append(list(circle_center))
                    # 判斷是頂部還是底部
                    position = self.coin_top_or_bottom(record)
                    if position == "top":
                        record["top_circle_center"].append(list(circle_center))
                    else:
                        record["bottom_circle_center"].append(list(circle_center))
                    record["radius"].append(radius)
                else:
                    # 如果是新硬幣，給它一個新的代號
                    label = f"C{len(self.coin_records) + 1}"
                    self.tracked_coins[circle_center] = (circle_center, label)
                    self.coin_records[label] = {
                        "frame_counter": self.frame_counter,
                        "appear_time": self.format_time(self.frame_counter, fps),
                        "disappear_time": self.format_time(self.frame_counter, fps),
                        "circle_center": [list(circle_center)],
                        "top_circle_center":  [],
                        "bottom_circle_center": [],
                        "radius": [radius],
                        "value": None,
                        "hight": None
                    }

                # 畫圓形輪廓
                cv2.circle(frame, circle_center, radius, (0, 255, 0), 2)
                # 標記圓心
                cv2.circle(frame, circle_center, 2, (0, 0, 255), 3)

                # 標記代號和位置 (C1-top, C2-bottom...)
                label_text = f"{label}-{position}"
                # 標記代號 (C1, C2, C3...)
                cv2.putText(frame, label_text, (i[0] + roi_x - 20, i[1] + roi_y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # 繪製ROI範圍的矩形框
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)
        return frame , self.coin_records

    def save_to_json(self, output_path):
        def convert_to_python_types(obj):
            """遞迴將 NumPy 型別轉換為 Python 標準型別"""
            if isinstance(obj, dict):
                return {k: convert_to_python_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_python_types(v) for v in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        # 將 coin_records 轉換為 Python 標準型別
        python_friendly_data = convert_to_python_types(self.coin_records)

        # 保存為 JSON
        with open(output_path, 'w') as json_file:
            json.dump(python_friendly_data, json_file, indent=4)

    def coin_value(self, coin_records):
        min_radius = 100
        max_radius = 0
        coin_radius_modes = []

        # 收集所有硬幣的半徑模式
        for record in coin_records.values():
            if record["radius"]:  # 確保有半徑資料
                coin_radius_mode = statistics.mode(record["radius"])  # 計算半徑的眾數
                coin_radius_modes.append(coin_radius_mode)
                min_radius = min(min_radius, coin_radius_mode)
                max_radius = max(max_radius, coin_radius_mode)

        # 計算半徑的區間範圍
        radius_rate = (max_radius - min_radius) / 4
        print(radius_rate)
        
        # 更新每個硬幣的數值
        for idx, coin_radius_mode in enumerate(coin_radius_modes):
            if min_radius <= coin_radius_mode < min_radius + radius_rate:
                coin_value = 1
            elif min_radius + radius_rate <= coin_radius_mode < min_radius + radius_rate * 2:
                coin_value = 5
            elif min_radius + radius_rate * 2 <= coin_radius_mode < min_radius + radius_rate * 3:
                coin_value = 10
            else:
                coin_value = 50

            # 更新對應硬幣的 "value"
            coin_records[list(coin_records.keys())[idx]]["value"] = coin_value
            print(f"Updated {list(coin_records.keys())[idx]} value to {coin_value} 圓")

    def coin_hight(self, coin_records):
        for key in coin_records.keys():
            y_coords = [point[1] for point in coin_records[key]["circle_center"]]
            
            min_y = min(y_coords)
            max_y = max(y_coords)
            
            coin_height = max_y - min_y
            coin_records[key]["hight"] = coin_height

def separate_foreground_background(frame, coin_records):
    # 創建與輸入幀大小相同的遮罩
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    for key, record in coin_records.items():
        # 確保 top_circle_center 列表不為空
        if not record["top_circle_center"]:
            continue
        # 計算方框的左上角和右下角座標點
        object_left_top_points = (record["top_circle_center"][-1][0] - record["radius"][-1], record["top_circle_center"][-1][1] - record["radius"][-1])
        object_right_bottom_points = (record["top_circle_center"][-1][0] + record["radius"][-1], record["top_circle_center"][-1][1] + record["radius"][-1]*2)

        # 在遮罩上繪製矩形方框，將框內設為前景 (白色)
        cv2.rectangle(mask, object_left_top_points, object_right_bottom_points, 255, -1)

        # 在輸入幀上顯示框
        cv2.rectangle(frame, object_left_top_points, object_right_bottom_points, (0, 255, 0), 2)
        print(f"物件框範圍: 左上角 {object_left_top_points}, 右下角 {object_right_bottom_points}")

    # 利用遮罩提取前景
    foreground = cv2.bitwise_and(frame, frame, mask=mask)

    # 創建背景 (遮罩取反)
    background_mask = cv2.bitwise_not(mask)
    background = cv2.bitwise_and(frame, frame, mask=background_mask)

    return foreground, background

def save_video(filename, fps=30, w=720, h=480):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(filename, fourcc, fps, (w, h))
    return out

if __name__ == "__main__":
    paused = False
    roi = (20, 110, 600, 400)  # ROI範圍
    video_path = '../video/transfer_video.mp4'
    output_path = 'data_test.json'
    
    out_coins_tracking_frame = save_video("coins_tracking_frame.mp4", fps=30)
    out_foreground = save_video("foreground.mp4", fps=30)
    out_background = save_video("background.mp4", fps=30)

    cap = cv2.VideoCapture(video_path)

    ret, prev_frame = cap.read()
    h, w = prev_frame.shape[:2]
    frame_counter = 0  # 當前幀計數

    catch_coin = CoinTracker(roi)

    while cap.isOpened():
        ret, frame = cap.read()
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not ret:
            break

        prep_frame = catch_coin.process_frame(frame)
        circles = catch_coin.find_circles(prep_frame)
        coins_tracking_frame, coin_records = catch_coin.coin_tracking(frame, circles, fps)
        foreground, background = separate_foreground_background(frame, coin_records)
        cv2.imshow("frame", frame)
        cv2.imshow("coins_tracking_frame", coins_tracking_frame)
        cv2.imshow("foreground", foreground)
        cv2.imshow("background", background)

        out_coins_tracking_frame.write(coins_tracking_frame)
        out_foreground.write(foreground)
        out_background.write(background)

        frame_counter +=1  # 當前幀計數
    
    # 處理鍵盤事件
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):  # 按 'q' 退出
            break
        elif key == ord("p"):  # 按 'p' 暫停/繼續
            paused = not paused
            while paused:
                key_pause = cv2.waitKey(1) & 0xFF
                if key_pause == ord("p"):  # 再次按 'p' 以繼續
                    paused = False
    cap.release()
    cv2.destroyAllWindows()
    
    # print("coin_records", coin_records)
    # 計算硬幣
    coin_value = catch_coin.coin_value(coin_records)
    catch_coin.coin_hight(coin_records)
    # 紀錄結果
    catch_coin.save_to_json(output_path)