import pyrealsense2 as rs
import cv2
import numpy as np
import time
import torch
from ultralytics import YOLO
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from PIL import Image
import mediapipe as mp
import re

class RealSenseVLM_ASFMNavigator:
    def __init__(self, yolo_model="yolov8n.pt", vlm_model="Salesforce/instructblip-vicuna-7b"):
        # RealSense 설정
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.pipeline.start(config)

        # 모델 설정
        self.yolo = YOLO(yolo_model).to('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = InstructBlipProcessor.from_pretrained(vlm_model)
        self.vlm_model = InstructBlipForConditionalGeneration.from_pretrained(vlm_model).to('cuda' if torch.cuda.is_available() else 'cpu')

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2)

        # 파라미터
        self.max_speed = 1.5
        self.min_speed = 0.2
        self.social_force_gain = 2.5
        self.personal_space_radius = 1.2  # meters

    def deproject_pixel_to_point(self, depth_frame, pixel):
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        depth = depth_frame.get_distance(int(pixel[0]), int(pixel[1]))
        point = rs.rs2_deproject_pixel_to_point(depth_intrin, [pixel[0], pixel[1]], depth)
        return np.array(point)  # [x, y, z]

    def detect_social_entities(self, color_frame, depth_frame):
        rgb_image = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
        image_height, image_width = color_frame.shape[:2]

        detected_info = {
            'humans': [],
            'has_gesture': False,
            'human_count': 0
        }

        # 손 제스처
        hands_result = self.hands.process(rgb_image)
        if hands_result.multi_hand_landmarks:
            for hand_lms in hands_result.multi_hand_landmarks:
                wrist = hand_lms.landmark[0]
                tips = [hand_lms.landmark[i] for i in [4, 8, 12, 16, 20]]
                if all(tip.y < wrist.y for tip in tips):
                    detected_info['has_gesture'] = True

        # YOLO 사람 탐지
        results = self.yolo(color_frame)[0]
        for box in results.boxes:
            if int(box.cls[0]) == 0:  # person 클래스
                bbox = box.xyxy[0].cpu().numpy()
                x_center = int((bbox[0] + bbox[2]) / 2)
                y_center = int((bbox[1] + bbox[3]) / 2)
                pos_3d = self.deproject_pixel_to_point(depth_frame, (x_center, y_center))
                if pos_3d[2] > 0:
                    detected_info['humans'].append({
                        'position': np.array(pos_3d),
                        'bbox': bbox
                    })

        detected_info['human_count'] = len(detected_info['humans'])
        return detected_info

    def calculate_social_force(self, robot_pos, humans):
        total_force = np.zeros(3)
        for human in humans:
            human_pos = human['position']
            diff = robot_pos - human_pos
            dist = np.linalg.norm(diff)
            if dist < self.personal_space_radius:
                force = self.social_force_gain * np.exp(-dist) * (diff / (dist + 1e-6))
                total_force += force
        return total_force

    def get_vlm_speed(self, color_frame, detected_info):
        min_dist = float('inf')
        for human in detected_info['humans']:
            dist = np.linalg.norm(human['position'])
            if dist < min_dist:
                min_dist = dist

        prompt = (
            f"There are {detected_info['human_count']} people detected. "
            f"The closest person is {min_dist:.2f} meters away. "
            f"Recommend a safe forward speed between {self.min_speed}m/s and {self.max_speed}m/s. "
            f"Answer only in format: [number]m/s."
        )

        inputs = self.processor(images=Image.fromarray(cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)),
                                 text=prompt, return_tensors="pt").to(self.device)
        outputs = self.vlm_model.generate(**inputs, max_new_tokens=10)
        decoded = self.processor.decode(outputs[0], skip_special_tokens=True)

        match = re.search(r"(\d+(\.\d+)?)\s*m/s", decoded)
        if match:
            return float(match.group(1))
        return self.max_speed  # fallback
    
    def visualize_results(self, color_image, detected_info, command_text):
        viz_image = color_image.copy()

        # 사람들 Bounding Box 그리기
        for human in detected_info['humans']:
            x1, y1, x2, y2 = map(int, human['bbox'])
            cv2.rectangle(viz_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 명령어(Command) 텍스트 표시
        cv2.putText(viz_image, f"Command: {command_text}",
                    (10, viz_image.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # 사람 수 표시
        cv2.putText(viz_image, f"Humans detected: {len(detected_info['humans'])}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if detected_info['has_gesture']:
            cv2.putText(viz_image, "GESTURE DETECTED",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return viz_image

    def run(self):
        try:
            while True:
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())

                detected_info = self.detect_social_entities(color_image, depth_frame)

                if detected_info['has_gesture']:
                    command_text = "STOP (Gesture Detected)"
                else:
                    robot_pos = np.array([0, 0, 0])
                    social_force = self.calculate_social_force(robot_pos, detected_info['humans'])
                    vlm_speed = self.get_vlm_speed(color_image, detected_info)

                    force_mag = np.linalg.norm(social_force)
                    adjusted_speed = np.clip(vlm_speed * (1 - force_mag), self.min_speed, self.max_speed)

                    if force_mag > 1e-3:
                        direction_angle = np.arctan2(social_force[0], social_force[2]) * 180 / np.pi
                        if direction_angle < -30:
                            direction = "Move Left"
                        elif direction_angle > 30:
                            direction = "Move Right"
                        else:
                            direction = "Move Straight"
                    else:
                        direction = "Move Straight"

                    command_text = f"{direction} at {adjusted_speed:.2f}m/s"

                # --- [화면에 표시] ---
                display_image = self.visualize_results(color_image, detected_info, command_text)
                cv2.imshow("RealSense Navigation", display_image)

                # --- [터미널에도 출력] ---
                print("[Command]", command_text)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    navigator = RealSenseVLM_ASFMNavigator()
    navigator.run()