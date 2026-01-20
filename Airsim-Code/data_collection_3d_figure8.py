import airsim
import time
import csv
import math
import threading
import os
import queue
import random

# --- 1. Initialization ---
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# --- 2. File and Data Setup ---
filename = "figure8_3d_dataset.csv"
print(f"Data will be saved to: {filename}")

with open(filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Timestamp", "Vx (m/s)", "Vy (m/s)", "Vz (m/s)", "Power (W)", "Altitude (m)"])

data_queue = queue.Queue()
recording = True

def record_data():
    while recording or not data_queue.empty():
        try:
            data = data_queue.get(timeout=1)
            with open(filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(data)
        except queue.Empty:
            continue

recording_thread = threading.Thread(target=record_data, daemon=True)
recording_thread.start()

# --- 3. Helper Functions ---

def generate_3d_figure8_path(scale_x, scale_y, scale_z, vh_speed, duration, center_x=0, center_y=0, start_z=-5.0, points_per_sec=10):
    path = []
    num_points = int(duration * points_per_sec)
    dt = 1.0 / points_per_sec
    
    approx_perimeter = 4.0 * max(scale_x, scale_y)
    if vh_speed < 0.1: vh_speed = 0.1
    time_per_cycle = approx_perimeter / vh_speed
    omega = 2 * math.pi / time_per_cycle

    for i in range(num_points + 1):
        t_sim = i * dt
        theta = omega * t_sim
        
        x = center_x + scale_x * math.sin(theta)
        y = center_y + scale_y * math.sin(2 * theta) / 2.0 
        z = start_z + scale_z * math.cos(theta)
        
        path.append(airsim.Vector3r(x, y, z))
        
    return path

def simulate_power(rotor_states):
    voltage = 15.2 + 0.3 * (2 * random.random() - 1)
    try:
        current = 0.0
        for rotor in rotor_states.rotors:
            speed = rotor['speed']
            speed = min(speed, 900)
            effective_speed = speed * (1 + random.uniform(-0.05, 0.05))
            current += (effective_speed ** 3) * 0.75e-7
    except:
        current = 0.0
    efficiency = 0.8 + random.uniform(-0.02, 0.02)
    return voltage * current * efficiency

# --- 4. Main Execution ---
sample_freq = 100 
sample_interval = 1 / sample_freq
total_missions = 200 

try:
    for mission_id in range(1, total_missions + 1):
        print(f"\nStarting Mission {mission_id}/{total_missions}...")

        client.reset()
        client.enableApiControl(True)
        client.armDisarm(True)
        client.takeoffAsync().join()
        
        client.hoverAsync().join()
        time.sleep(1)
        
        start_state = client.getMultirotorState().kinematics_estimated.position
        center_z = start_state.z_val 
        center_x = start_state.x_val
        center_y = start_state.y_val

        client.simSetWind(airsim.Vector3r(random.uniform(-1.5, 1.5), random.uniform(-1.5, 1.5), 0))
        
        vh_speed = random.uniform(3.0, 10.0)
        scale_x = random.uniform(30.0, 60.0) 
        scale_y = random.uniform(30.0, 60.0)
        scale_z = random.uniform(5.0, 15.0)
        duration_mission = random.uniform(40.0, 80.0) 

        approx_perimeter = 4.0 * max(scale_x, scale_y)
        omega = (2 * math.pi) / (approx_perimeter / vh_speed)
        max_vv = scale_z * omega

        print(f"Target: Vh~{vh_speed:.1f}m/s, Max Vv~{max_vv:.1f}m/s")
        print(f"Dims: X={scale_x:.0f}m, Y={scale_y:.0f}m, Z_amp={scale_z:.0f}m")

        path = generate_3d_figure8_path(scale_x, scale_y, scale_z, vh_speed, duration_mission, 
                                        center_x, center_y, center_z)

        client.moveOnPathAsync(path, 
                             velocity=vh_speed * 1.5,
                             drivetrain=airsim.DrivetrainType.ForwardOnly,
                             yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0))

        start_time = time.time()
        while time.time() - start_time < duration_mission:
            state = client.getMultirotorState()
            kin = state.kinematics_estimated
            
            if state.collision.has_collided:
                print("Collision detected!")
                break
            
            power = simulate_power(client.getRotorStates())
            
            data_queue.put([
                time.time(), 
                kin.linear_velocity.x_val, 
                kin.linear_velocity.y_val,
                -kin.linear_velocity.z_val, 
                power, 
                -kin.position.z_val
            ])
            
            if not state.landed_state == airsim.LandedState.Flying:
                 break
            time.sleep(sample_interval)

        client.hoverAsync().join()
        client.landAsync().join()
        client.armDisarm(False)
        client.enableApiControl(False)
        print(f"Mission {mission_id} complete")

except KeyboardInterrupt:
    print("Stopping...")
finally:
    recording = False
    recording_thread.join()
    client.reset()
    client.enableApiControl(False)
