import cv2
import numpy as np
import json


class CoinTracker:
    def __init__(self, video_path, output_path, roi, min_radius=10, max_radius=25, area_threshold_min=2200, area_threshold_max=5000):
        self.video_path = video_path
        self.output_path = output_path
        self.roi = roi
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.json_output_path = "data.json"
        self.area_threshold_min = area_threshold_min
        self.area_threshold_max = area_threshold_max
        self.tracked_coins = {}  # 用於追蹤硬幣的ID和其位置
        self.coin_records = {}  # 用於記錄硬幣的詳細信息
        self.frame_counter = 0  # 當前幀計數
        self.paused = False  # 控制播放/暫停的變數

        # 初始化影片讀取和寫入
        self.cap = cv2.VideoCapture(self.video_path)
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.out = cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    def find_matching_coin(self, new_center, threshold=40):
        """搜尋是否有與新檢測到的圓心相近的硬幣"""
        for key, (prev_center, label) in self.tracked_coins.items():
            if np.linalg.norm(np.array(new_center) - np.array(prev_center)) < threshold:
                return key, label  # 找到匹配的圓形，返回其ID
        return None, None  # 沒有找到匹配的圓形

    def format_time(self, milliseconds, fps):
        """將幀數轉換為時間格式 (MM:SS:MMM)"""
        seconds = milliseconds // fps
        milliseconds = int((milliseconds % fps) * (1000 / fps))
        return f"{seconds // 60}:{seconds % 60}:{milliseconds}"
    
    def save_to_json(self):
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
        with open(self.json_output_path, 'w') as json_file:
            json.dump(python_friendly_data, json_file, indent=4)

    def process_frame(self, frame):
        output = frame.copy()
        roi_x, roi_y, roi_w, roi_h = self.roi
        roi = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

        # 更新當前幀計數
        self.frame_counter += 1
        fps = self.cap.get(cv2.CAP_PROP_FPS)

        # 轉換為灰階圖像
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        gray_blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        gray_blurred = 255 - gray_blurred

        # 使用霍夫變換來檢測圓形
        circles = cv2.HoughCircles(
            gray_blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=50,
            param1=50,
            param2=30,
            minRadius=self.min_radius,
            maxRadius=self.max_radius
        )

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
                else:
                    # 如果是新硬幣，給它一個新的代號
                    label = f"C{len(self.coin_records) + 1}"
                    self.tracked_coins[circle_center] = (circle_center, label)
                    self.coin_records[label] = {
                        "frame_counter": self.frame_counter,
                        "appear_time": self.format_time(self.frame_counter, fps),
                        "disappear_time": self.format_time(self.frame_counter, fps),
                        "circle_center": [list(circle_center)],
                        "radius": radius
                    }

                # 畫圓形輪廓
                cv2.circle(output, circle_center, radius, (0, 255, 0), 2)
                # 標記圓心
                cv2.circle(output, circle_center, 2, (0, 0, 255), 3)

                # 標記代號 (C1, C2, C3...)
                cv2.putText(output, label, (i[0] + roi_x - 20, i[1] + roi_y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # 繪製ROI範圍的矩形框
        cv2.rectangle(output, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)
        return output

    def run(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            output = self.process_frame(frame)

            # 寫入處理後的幀到輸出影片
            self.out.write(output)

            # 顯示結果
            cv2.imshow('Detected Coins with Labels', output)

            # 檢測按鍵，按 'q' 退出，按 'p' 暫停/繼續
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # 按 'q' 退出
                break
            elif key == ord('p'):  # 按 'p' 暫停/繼續
                self.paused = not self.paused
                while self.paused:
                    key_pause = cv2.waitKey(1) & 0xFF
                    if key_pause == ord('p'):  # 按 'p' 恢復播放
                        self.paused = False

        print(self.coin_records) # 輸出硬幣的追蹤資料
        self.save_to_json()
        # 釋放資源
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    roi = (20, 110, 600, 400)  # ROI範圍
    video_path = '../video/transfer_video.mp4'
    output_path = 'catch_coin_video.mp4'

    tracker = CoinTracker(video_path, output_path, roi)
    tracker.run()
