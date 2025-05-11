import pyrealsense2 as rs
import numpy as np
import cv2
import time
import torch
from ultralytics import YOLO
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from PIL import Image
import re
import matplotlib.pyplot as plt

class ASFMNavigatorSim:
    def __init__(self):
        self.yolo = YOLO('yolov8n.pt').to('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
        self.vlm_model = InstructBlipForConditionalGeneration.from_pretrained(
            "Salesforce/instructblip-vicuna-7b"
        ).to('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.A = 2.1
        self.B = 0.3
        self.C = 0.0
        self.personal_space = 1.2
        self.min_speed = 0.2
        self.max_speed = 1.5

        self.last_human_positions = {}

        # 시뮬레이션 데이터 기록용
        self.simulation_time = []
        self.simulation_speed = []

    def simulate_environment(self, steps=100):
        current_pos = np.array([0.0, 0.0, 0.0])

        for step in range(steps):
            time.sleep(0.05)  # 시뮬레이션 시간 간격

            # 가상의 인간 데이터를 생성 (거리, 속도 랜덤)
            human_pos = np.random.uniform(low=0.5, high=3.0, size=3)
            human_vel = np.random.uniform(low=-0.5, high=0.5, size=3)
            humans = [{'position': human_pos, 'velocity': human_vel}]

            # ASFM Force 계산
            force = self.compute_asfm_force(current_pos, humans)
            speed = self.compute_adaptive_speed(humans)

            # 명령어 시뮬레이션
            move_direction = force / (np.linalg.norm(force) + 1e-6) if np.linalg.norm(force) > 1e-3 else np.array([0, 0, 1])

            # 결과 기록
            self.simulation_time.append(step * 0.05)
            self.simulation_speed.append(speed)

            print(f"[Step {step}] Direction: {move_direction}, Speed: {speed:.2f} m/s")

        self.visualize_simulation()

    def compute_adaptive_speed(self, humans):
        if not humans:
            return self.max_speed
        min_speed = self.max_speed
        for human in humans:
            dist = np.linalg.norm(human['position'])
            vel = human.get('velocity', np.zeros(3))
            approaching = np.dot(vel, human['position']) < 0

            if dist < 0.3:
                speed = 0.0
            elif dist < 2.0:
                ratio = (dist - 0.3) / (2.0 - 0.3)
                speed = self.min_speed + ratio * (self.max_speed - self.min_speed)
                if approaching:
                    speed *= 0.7
            else:
                speed = self.max_speed

            min_speed = min(min_speed, speed)
        return min_speed

    def compute_asfm_force(self, robot_pos, humans):
        total_force = np.zeros(3)
        for human in humans:
            diff = robot_pos - human['position']
            dist = np.linalg.norm(diff)
            if dist < self.personal_space:
                static_force = self.A * np.exp(self.B * dist + self.C)
                moving_force = self.A * np.exp((dist - 0.5) * self.B + self.C)
                direction = diff / (dist + 1e-6)
                total_force += (static_force + moving_force) * direction
        return total_force

    def visualize_simulation(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.simulation_time, self.simulation_speed, marker='o')
        plt.title('Speed Change Over Time (Simulation)')
        plt.xlabel('Time (s)')
        plt.ylabel('Speed (m/s)')
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    navigator = ASFMNavigatorSim()
    navigator.simulate_environment(steps=100)
