import cv2
import numpy as np
import time

"""
gtp 以v3 產生將createBackgroundSubtractorMOG2 與 v3 結合
"""

class OpticalFlowWithDepth:
    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16)
        self.paused = False

    def calculate_flow_length(self, flow):
        return np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

    def calculate_foe(self, flow, D_mid):
        m = np.linalg.norm(flow, axis=2)
        z = np.where((m == 0) | (D_mid == 0), 0, D_mid / m)
        return z

    def draw_optical_flow(self, frame, flow, step=16):
        h, w = frame.shape[:2]
        for y in range(0, h, step):
            for x in range(0, w, step):
                fx, fy = flow[y, x]
                end_point = (int(x + fx), int(y + fy))
                magnitude = np.sqrt(fx**2 + fy**2)
                color = (int(min(magnitude * 10, 255)), 255 - int(min(magnitude * 10, 255)), 0)
                cv2.arrowedLine(frame, (x, y), end_point, color, 1, tipLength=0.3)

    def process_frame(self, frame, prev_gray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        fg_mask = self.bg_subtractor.apply(gray)
        fg_mask = cv2.medianBlur(fg_mask, 5)
        D_mid = fg_mask.astype(float) / 255.0
        z = self.calculate_foe(flow, D_mid)
        z_resized = cv2.resize(z, (frame.shape[1], frame.shape[0]))
        return gray, flow, z_resized

    def main(self, video_path):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if not ret:
            print("Failed to open video.")
            return

        prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        while True:
            if not self.paused:
                start_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    print("End of video.")
                    break

                prev_gray, flow, z_resized = self.process_frame(frame, prev_gray)

                # Draw optical flow on the frame
                self.draw_optical_flow(frame, flow)
                
                # Display depth map
                depth_map = (z_resized / z_resized.max() * 255).astype(np.uint8)
                depth_map_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
                
                # Show frames
                cv2.imshow("Optical Flow", frame)
                cv2.imshow("Depth Map", depth_map_colored)
                print(f"Frame processing time: {time.time() - start_time:.2f} seconds")
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                self.paused = not self.paused

                if self.paused:
                    print("Paused.")
                    while self.paused:
                        cv2.putText(frame, "Paused", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.imshow("Optical Flow", frame)
                        if cv2.waitKey(1) & 0xFF == ord('p'):
                            self.paused = False
                            print("Resumed.")

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "../kitti/kitti.mp4"  # Replace with your video file path
    optical_flow_with_depth = OpticalFlowWithDepth()
    optical_flow_with_depth.main(video_path)
