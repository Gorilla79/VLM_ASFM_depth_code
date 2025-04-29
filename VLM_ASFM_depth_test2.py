import pyrealsense2 as rs
import numpy as np
import cv2
import time
import torch
import re
from PIL import Image
from ultralytics import YOLO
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

class RealSenseVLM_ASFMNavigator:
    def __init__(self, yolo_model="yolov8n.pt", vlm_model="Salesforce/instructblip-vicuna-7b"):
        # RealSense 설정
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.pipeline.start(config)

        # 모델 불러오기
        self.yolo = YOLO(yolo_model).to('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = InstructBlipProcessor.from_pretrained(vlm_model)
        self.vlm_model = InstructBlipForConditionalGeneration.from_pretrained(vlm_model).to('cuda' if torch.cuda.is_available() else 'cpu')

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 파라미터
        self.max_speed = 1.5
        self.min_speed = 0.2
        self.personal_space = 1.2  # meters
        self.social_force_gain = 2.5
        
        self.last_vlm_query_time = 0
        self.vlm_query_interval = 2.0  # VLM 호출 최소 간격 (초)

    def deproject(self, depth_frame, pixel):
        """RGB pixel 좌표를 3D 위치로 변환"""
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        depth = depth_frame.get_distance(int(pixel[0]), int(pixel[1]))
        if depth == 0:
            return None
        point = rs.rs2_deproject_pixel_to_point(depth_intrin, [pixel[0], pixel[1]], depth)
        return np.array(point)

    def detect_humans(self, color_frame, depth_frame):
        """YOLO로 사람 탐지 + Depth 기반 3D 위치 추출"""
        results = self.yolo(color_frame)[0]
        humans = []
        for box in results.boxes:
            if int(box.cls[0]) == 0:  # 'person' 클래스
                bbox = box.xyxy[0].cpu().numpy()
                cx = int((bbox[0] + bbox[2]) / 2)
                cy = int((bbox[1] + bbox[3]) / 2)
                pos_3d = self.deproject(depth_frame, (cx, cy))
                if pos_3d is not None and pos_3d[2] > 0:
                    humans.append({'position': pos_3d, 'bbox': bbox})
        return humans

    def calculate_social_force(self, robot_pos, humans):
        """ASFM 스타일 사회적 힘 계산"""
        total_force = np.zeros(3)
        for human in humans:
            diff = robot_pos - human['position']
            dist = np.linalg.norm(diff)
            if dist < self.personal_space:
                force = self.social_force_gain * np.exp(-dist) * (diff / (dist + 1e-6))
                total_force += force
        return total_force

    def generate_prompt(self, color_frame, humans):
        """VLM에 입력할 상황 설명 프롬프트 생성"""
        min_dist = min([np.linalg.norm(h['position']) for h in humans], default=10.0)
        human_count = len(humans)
        prompt = (
            f"There are {human_count} people detected. "
            f"The closest person is {min_dist:.2f} meters away. "
            f"The robot needs to move safely in a crowded environment. "
            f"Decide appropriate action and safe speed (0.2m/s ~ 1.5m/s). "
            f"Answer format: [Action] [Speed]m/s. "
            f"Actions: Move_Left, Move_Right, Move_Straight, Stop"
        )
        return prompt

    def query_vlm(self, color_frame, humans):
        """VLM을 통해 행동과 속도 결정"""
        prompt = self.generate_prompt(color_frame, humans)
        inputs = self.processor(images=Image.fromarray(cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)),
                                 text=prompt, return_tensors="pt").to(self.device)
        outputs = self.vlm_model.generate(**inputs, max_new_tokens=20)
        decoded = self.processor.decode(outputs[0], skip_special_tokens=True)
        print("[VLM Response]", decoded)

        match = re.search(r"(Move_Left|Move_Right|Move_Straight|Stop)\s+(\d+(\.\d+)?)m/s", decoded)
        if match:
            action = match.group(1)
            speed = float(match.group(2))
            return action, speed
        else:
            return "Move_Straight", 1.0  # 기본값

    def visualize(self, frame, humans, command_text):
        """시각화"""
        for human in humans:
            x1, y1, x2, y2 = map(int, human['bbox'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(frame, f"Command: {command_text}", (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Humans: {len(humans)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return frame

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

                humans = self.detect_humans(color_image, depth_frame)
                robot_pos = np.array([0, 0, 0])

                current_time = time.time()

                # 군중 밀집도 평가
                if len(humans) >= 3 and (current_time - self.last_vlm_query_time > self.vlm_query_interval):
                    action, speed = self.query_vlm(color_image, humans)
                    self.last_vlm_query_time = current_time
                else:
                    social_force = self.calculate_social_force(robot_pos, humans)
                    force_mag = np.linalg.norm(social_force)
                    if force_mag > 1e-3:
                        angle = np.arctan2(social_force[0], social_force[2]) * 180 / np.pi
                        if angle < -30:
                            action = "Move_Left"
                        elif angle > 30:
                            action = "Move_Right"
                        else:
                            action = "Move_Straight"
                    else:
                        action = "Move_Straight"
                    speed = np.clip(self.max_speed * (1 - force_mag), self.min_speed, self.max_speed)

                command_text = f"{action} at {speed:.2f}m/s"
                print("[Command]", command_text)

                # 화면에 표시
                display_image = self.visualize(color_image, humans, command_text)
                cv2.imshow("RealSense VLM+ASFM Navigation", display_image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    navigator = RealSenseVLM_ASFMNavigator()
    navigator.run()