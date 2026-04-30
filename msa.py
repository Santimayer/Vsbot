import machine
import network
import socket
import time
import json
import math

class SpatialPIDController:
    def __init__(self, kp, ki, kd, max_out):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_out = max_out
        self.integral = 0.0
        self.previous_error = 0.0
        self.last_time = time.ticks_us()

    def compute_correction(self, error):
        current_time = time.ticks_us()
        dt = time.ticks_diff(current_time, self.last_time) / 1000000.0
        if dt <= 0.0:
            dt = 0.001
            
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt
        
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        
        self.previous_error = error
        self.last_time = current_time
        
        if output > self.max_out:
            output = self.max_out
        elif output < -self.max_out:
            output = -self.max_out
            
        return output

class DualAxisKinematicActuator:
    def __init__(self, pin_in1, pin_in2, pin_ena, pin_in3, pin_in4, pin_enb, pwm_freq=1000):
        self.motor_x_fwd = machine.Pin(pin_in1, machine.Pin.OUT)
        self.motor_x_rev = machine.Pin(pin_in2, machine.Pin.OUT)
        self.pwm_drive_x = machine.PWM(machine.Pin(pin_ena), freq=pwm_freq)
        
        self.motor_y_fwd = machine.Pin(pin_in3, machine.Pin.OUT)
        self.motor_y_rev = machine.Pin(pin_in4, machine.Pin.OUT)
        self.pwm_drive_y = machine.PWM(machine.Pin(pin_enb), freq=pwm_freq)
        
        self.pid_x = SpatialPIDController(3.5, 0.1, 1.2, 1023)
        self.pid_y = SpatialPIDController(3.5, 0.1, 1.2, 1023)
        
        self.optical_deadzone_radius = 25.0
        self.actuation_threshold_confidence = 0.65

    def execute_spatial_transform(self, dx_pixel_error, dy_pixel_error, tracking_confidence):
        if tracking_confidence < self.actuation_threshold_confidence:
            self.engage_emergency_brake()
            return False

        if abs(dx_pixel_error) > self.optical_deadzone_radius:
            velocity_vector_x = self.pid_x.compute_correction(dx_pixel_error)
            self.drive_azimuth_axis(velocity_vector_x)
        else:
            self.drive_azimuth_axis(0)

        if abs(dy_pixel_error) > self.optical_deadzone_radius:
            velocity_vector_y = self.pid_y.compute_correction(dy_pixel_error)
            self.drive_elevation_axis(velocity_vector_y)
        else:
            self.drive_elevation_axis(0)
            
        return True

    def drive_azimuth_axis(self, velocity_magnitude):
        duty_cycle = int(abs(velocity_magnitude))
        if duty_cycle > 1023:
            duty_cycle = 1023
            
        self.pwm_drive_x.duty(duty_cycle)
        
        if velocity_magnitude > 0:
            self.motor_x_fwd.value(1)
            self.motor_x_rev.value(0)
        elif velocity_magnitude < 0:
            self.motor_x_fwd.value(0)
            self.motor_x_rev.value(1)
        else:
            self.motor_x_fwd.value(0)
            self.motor_x_rev.value(0)

    def drive_elevation_axis(self, velocity_magnitude):
        duty_cycle = int(abs(velocity_magnitude))
        if duty_cycle > 1023:
            duty_cycle = 1023
            
        self.pwm_drive_y.duty(duty_cycle)
        
        if velocity_magnitude > 0:
            self.motor_y_fwd.value(1)
            self.motor_y_rev.value(0)
        elif velocity_magnitude < 0:
            self.motor_y_fwd.value(0)
            self.motor_y_rev.value(1)
        else:
            self.motor_y_fwd.value(0)
            self.motor_y_rev.value(0)

    def engage_emergency_brake(self):
        self.drive_azimuth_axis(0)
        self.drive_elevation_axis(0)

class BoundingBoxTelemetry:
    def __init__(self, raw_network_payload):
        self.is_valid_frame = False
        self.roi_x = 0
        self.roi_y = 0
        self.roi_width = 0
        self.roi_height = 0
        self.optical_center_x = 0.0
        self.optical_center_y = 0.0
        self.frame_resolution_w = 0
        self.frame_resolution_h = 0
        self.detection_confidence = 0.0
        self.neural_class_id = -1
        self.decode_inference_payload(raw_network_payload)

    def decode_inference_payload(self, payload):
        try:
            inference_data = json.loads(payload)
            self.roi_x = int(inference_data.get("bounding_box_x", 0))
            self.roi_y = int(inference_data.get("bounding_box_y", 0))
            self.roi_width = int(inference_data.get("bounding_box_w", 0))
            self.roi_height = int(inference_data.get("bounding_box_h", 0))
            self.frame_resolution_w = int(inference_data.get("camera_res_w", 640))
            self.frame_resolution_h = int(inference_data.get("camera_res_h", 480))
            self.detection_confidence = float(inference_data.get("inference_score", 0.0))
            self.neural_class_id = int(inference_data.get("target_class_id", -1))

            self.optical_center_x = self.roi_x + (self.roi_width / 2.0)
            self.optical_center_y = self.roi_y + (self.roi_height / 2.0)

            if self.frame_resolution_w > 0 and self.frame_resolution_h > 0 and self.roi_width > 0:
                self.is_valid_frame = True
        except:
            self.is_valid_frame = False

    def calculate_focal_displacement(self):
        focal_origin_x = self.frame_resolution_w / 2.0
        focal_origin_y = self.frame_resolution_h / 2.0
        
        delta_pixel_x = self.optical_center_x - focal_origin_x
        delta_pixel_y = self.optical_center_y - focal_origin_y
        
        return delta_pixel_x, delta_pixel_y

class UDPOpticalFlowReceiver:
    def __init__(self, bind_ip, bind_port):
        self.ip_address = bind_ip
        self.port = bind_port
        self.telemetry_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.telemetry_socket.bind((self.ip_address, self.port))
        self.telemetry_socket.setblocking(False)

    def fetch_inference_packet(self):
        try:
            packet_data, client_addr = self.telemetry_socket.recvfrom(2048)
            return packet_data.decode('utf-8')
        except OSError:
            return None

def establish_wireless_uplink(ssid_target, psk_target):
    wlan_interface = network.WLAN(network.STA_IF)
    wlan_interface.active(True)
    if not wlan_interface.isconnected():
        wlan_interface.connect(ssid_target, psk_target)
        connection_timeout = 0
        while not wlan_interface.isconnected() and connection_timeout < 30:
            time.sleep(0.5)
            connection_timeout += 1
    if wlan_interface.isconnected():
        return wlan_interface.ifconfig()[0]
    return None

def main_tracking_loop():
    local_ipv4 = establish_wireless_uplink("VISION_NET_SSID", "VISION_NET_KEY")
    if not local_ipv4:
        machine.reset()

    telemetry_stream = UDPOpticalFlowReceiver("0.0.0.0", 8080)
    
    kinematic_platform = DualAxisKinematicActuator(
        pin_in1=12, pin_in2=14, pin_ena=13, 
        pin_in3=27, pin_in4=26, pin_enb=25
    )

    system_armed = True
    timestamp_last_frame = time.ticks_ms()
    frame_timeout_ms = 1500
    valid_target_classes = [0, 2, 7] 

    while system_armed:
        raw_inference_stream = telemetry_stream.fetch_inference_packet()

        if raw_inference_stream:
            timestamp_last_frame = time.ticks_ms()
            frame_metrics = BoundingBoxTelemetry(raw_inference_stream)

            if frame_metrics.is_valid_frame:
                if frame_metrics.neural_class_id in valid_target_classes:
                    displacement_x, displacement_y = frame_metrics.calculate_focal_displacement()
                    
                    kinematic_platform.execute_spatial_transform(
                        displacement_x, 
                        displacement_y, 
                        frame_metrics.detection_confidence
                    )
                else:
                    kinematic_platform.engage_emergency_brake()
            else:
                pass
        else:
            current_timestamp = time.ticks_ms()
            if time.ticks_diff(current_timestamp, timestamp_last_frame) > frame_timeout_ms:
                kinematic_platform.engage_emergency_brake()

        time.sleep_ms(5)

if __name__ == '__main__':
    main_tracking_loop()
