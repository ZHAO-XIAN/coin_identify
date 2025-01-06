import cv2
import numpy as np
import json
import statistics
from scipy.stats import norm

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

    def coin_top_or_bottom(self, circle_center):
        """
        根據已記錄的圓心座標，判斷該圓是頂部還是底部
        :param circle_center: 當前檢測到的圓心座標
        :return: 'top' 或 'bottom'
        """
        x, y = circle_center

        closest_coin = None
        min_distance = float('inf')

        for key, (prev_center, label) in self.tracked_coins.items():
            # 計算距離差異（可以根據需要選擇合適的距離計算方法）
            distance = (x - prev_center[0])**2 + (y - prev_center[1])**2  # 使用平方的歐氏距離來計算

            # 找到最接近的圓心
            if distance < min_distance:
                min_distance = distance
                closest_coin = (prev_center, label)

        # 根據最接近的圓心判斷是頂部還是底部
        if closest_coin:
            prev_center, label = closest_coin
            if prev_center[1] > y:
                return "top"
            else:
                return "bottom"
        
        return "top"
        
    def coin_tracking(self, frame, circles , fps):
        frame = frame.copy()
        roi_x, roi_y, roi_w, roi_h = self.roi

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for idx, i in enumerate(circles[0, :]):
                circle_center = (i[0] + roi_x, i[1] + roi_y)
                radius = i[2]

                # 判斷是頂部還是底部
                position = self.coin_top_or_bottom(circle_center)

                # 尋找是否有匹配的硬幣
                matching_key, label = self.find_matching_coin(circle_center)

                if matching_key is not None:
                    # 如果找到了匹配的硬幣，更新位置並保持原來的代號
                    self.tracked_coins[matching_key] = (circle_center, label)
                    record = self.coin_records[label]
                    record["disappear_time"] = self.format_time(self.frame_counter, fps)
                    record["circle_center"].append(list(circle_center))
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
                        "top_circle_center": [list(circle_center)] if position == "top" else [],
                        "bottom_circle_center": [list(circle_center)] if position == "bottom" else [],
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
        for record in coin_records.values():
            coin_radius_mode = statistics.mode(record["radius"])
            min_radius = min(min_radius, coin_radius_mode)
            max_radius = max(max_radius, coin_radius_mode)

        radius_rate = (max_radius - min_radius)/4

        for idx, coin_radius_mode in enumerate(coin_radius_modes):
            if min_radius <= coin_radius_mode < min_radius + radius_rate:
                coin_value = 1
            elif min_radius + radius_rate <= coin_radius_mode < min_radius + radius_rate*2:
                coin_value = 5
            elif min_radius + radius_rate*2 <= coin_radius_mode < min_radius + radius_rate*3:
                coin_value = 10
            else:
                coin_value = 50
            
            coin_records[list(coin_records.keys())[idx]]["value"] = coin_value
            print(f"Updated {list(coin_records.keys())[idx]} value to {coin_value} 圓")

    def coin_hight(self, coin_records):
        for key in coin_records.keys():
            y_coords = [point[1] for point in coin_records[key]["circle_center"]]
            
            min_y = min(y_coords)
            max_y = max(y_coords)
            
            coin_height = max_y - min_y
            coin_records[key]["hight"] = coin_height

def separate_foreground_background(frame, circles, roi):
    roi_x, roi_y, roi_w, roi_h = roi
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(mask, (x + roi_x, y + roi_y), r, (255), thickness=-1)
    foreground = cv2.bitwise_and(frame, frame, mask=mask)
    mask = cv2.bitwise_not(mask)
    background = cv2.bitwise_and(frame, frame, mask=mask)
    return foreground, background

if __name__ == "__main__":
    paused = False
    roi = (20, 110, 600, 400)  # ROI範圍
    video_path = '../video/transfer_video.mp4'
    output_path = 'data_test.json'
    
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
        foreground, background = separate_foreground_background(frame, circles, roi)
        
        coins_tracking_frame, coin_records = catch_coin.coin_tracking(frame, circles, fps)
        cv2.imshow("frame", frame)
        cv2.imshow("coins_tracking_frame", coins_tracking_frame)
        cv2.imshow("foreground", foreground)
        cv2.imshow("background", background)

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
    
    print("coin_records", coin_records)
    # 計算硬幣
    coin_value = catch_coin.coin_value(coin_records)
    catch_coin.coin_hight(coin_records)
    # 紀錄結果
    catch_coin.save_to_json(output_path)
    # print("coin_records", coin_records)