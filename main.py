import cv2
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from tqdm import tqdm

cap = cv2.VideoCapture('videos/vid1.MOV')

if not cap.isOpened():
    print("Error opening video")

fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

frame_skip = 5

frames = []

for i in tqdm(range(0, frame_count, frame_skip), desc="Processing Frames", unit="frames"):
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

cap.release()

def butter_bandpass(freq_min, freq_max, fps, order=4):
    nyquist = 0.5 * fps
    print("butter_bandpass")
    print(f"Nyquist Frequency: {nyquist}")  # Debuggig
    if freq_max > nyquist:
        print(f"Warning: freq_max ({freq_max}) exceeds Nyquist ({nyquist}). Clamping to Nyquist.")
        freq_max = nyquist
    low = freq_min / nyquist
    high = freq_max / nyquist

    if high >= 1:
        print(f"Warning: high frequency is 1. Adjusting to be slightly less than 1.")
        high = 0.9999
        
    print(f"Low Frequency (normalized): {low}")  # Debugging
    print(f"High Frequency (normalized): {high}")  # Debugging
    
    if low <= 0 or high >= 1:
        raise ValueError("Frequency out of range")

    b, a = butter(order, [low, high], btype='band')
    return b, a

def eulerian_video_magnification(frames, freq_min=0.5, freq_max=20.0, amplification=50):
    print("eulerian_video_magnification")
    #Convert frames to numpy array
    frames = np.array(frames, dtype=np.float32)
    #Apply bandpass filter to the frames
    b, a = butter_bandpass(freq_min, freq_max, fps)
    filtered_frames = np.apply_along_axis(lambda x: filtfilt(b, a, x), 0, frames)
    amplified_frames = frames + filtered_frames * amplification
    return np.clip(amplified_frames, 0, 255).astype(np.uint8)

magnified_frames = eulerian_video_magnification(frames)

def optical_flow(frames):
    flow_magnitudes = []
    print("optical_flow")
    prev_gray = frames[0]    

    for i in range(1, len(frames)):
        next_gray = frames[i]
        flow = np.zeros_like(prev_gray, dtype=np.float32)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        avg_mag = np.mean(mag)
        flow_magnitudes.append(avg_mag)

        prev_gray = next_gray
    return flow_magnitudes

motion_data = optical_flow(magnified_frames)

def extract_frequencies(motion_data, fps):
    print("extract_frequencies")
    n = len(motion_data)
    time_step = 1.0/fps
    freqs = np.fft.fftfreq(n, time_step)
    fft_values = np.abs(np.fft.rfft(motion_data))

    #Get dominant frequency
    dominant_freq = freqs[np.argmax(fft_values)]
    return freqs, fft_values, dominant_freq

freqs, fft_values, dominant_freq = extract_frequencies(motion_data, fps)
print(f"Detected Vibration Frequency: {dominant_freq: .2f} Hz")

def overlay_motion(frames, motion_data):
    print("overlay_motion")
    overlayed_frames = []

    for i in range(len(frames) - 1):
        frame = cv2.cvtColor(frames[i], cv2.COLOR_GRAY2BGR)
        intensity = int(motion_data[i] * 500) #Scale for visibility
        color = (0, 0, min(intensity, 255)) # Use blue for vibration

        cv2.putText(frame, f"Vibration: {intensity}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        overlayed_frames.append(frame)
    return overlayed_frames

visulization_frames = overlay_motion(magnified_frames, motion_data)

out = cv2.VideoWriter("./output.mov", cv2.VideoWriter.fourcc(*'mp4v'), 30, (frame_width, frame_height))

for frame in tqdm(visulization_frames, desc="Writing Frames", unit="frames"):
    out.write(frame)

print("releasing")
out.release()
cv2.destroyAllWindows()
print("done")