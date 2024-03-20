import websocket
import json
import threading
import numpy as np
import cv2
import pdb

class SensorSocket:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.lastmessage = {"message":"No message"}
        self.wsapp = websocket.WebSocketApp("ws://{}:{}".format(ip, port), on_message=self.on_message)
        # self.threader(self.mesreader, name="Receiver")
        self.threader(self.wsapp.run_forever, name="WebSocket")
        # self.wsapp.run_forever()  # Start listening for messages
        self.sensor_reading = None

    def on_message(self, wsapp, message):
        try:
            data = json.loads(message)
        except Exception:
            pass
        else:
            try:
                if data["message"] != "Welcome":
                    self.lastmessage = data
            except Exception:
                pass

    def threader(self, target, args=False, **targs):
        if args:
            targs["args"] = (args,)
        thr = threading.Thread(target=target, **targs)
        thr.daemon = True
        thr.start()
    
    def message_parser(self, msg_obj):
        processed_data = []
        for sensor_idx in range(msg_obj['sensors']):
            forces = []
            sensor_data = msg_obj.get(str(sensor_idx + 1))
            if sensor_data is None:
                print(f"Sensor data not found for index {sensor_idx + 1}")
                return None
            taxels = sensor_data.get("taxels")
            data_val = sensor_data.get("data").split(",")
            forces = sensor_data.get("calibrated")
            if len(data_val) != 3 * taxels:
                print(f"Taxel count mismatch: {taxels} != {int(len(data_val) / 3)}")
                return None

            forces_array = np.zeros((4, 4, 3))
            t_idx = 0 
            for i in range(0, len(forces), 3):
                x, y, z = forces[i:i + 3]
                x = np.clip(x, 0.5, 10) - 0.5
                y = np.clip(y, 0.5, 10) - 0.5
                z = np.clip(z, 0.5, 10) - 0.5
                forces_array[t_idx // 4, t_idx % 4, :] = [x, y, z]
                t_idx+=1
            
            processed_data.append(forces_array)
        return processed_data
    
    def visualize_sensor_data(self, sensor_data):
        # Create a blank image
        image_size = 400
        img = np.zeros((image_size, image_size, 3), np.uint8)
        # Define grid parameters
        grid_size = 4
        cell_size = image_size // grid_size
        # Scale factor for visualization
        scale_factor = 10
        # Loop through each cell in the grid
        for row in range(grid_size):
            for col in range(grid_size):
                # Get forces for the current cell
                forces = sensor_data[row, col]
                # Calculate circle center position
                center_x = col * cell_size + cell_size // 2
                center_y = row * cell_size + cell_size // 2
                # Calculate displacements for visualization
                disp_x = int(forces[0] * scale_factor)  # X displacement
                disp_y = int(forces[1] * scale_factor)  # Y displacement
                # Calculate circle radius based on Z force
                radius = np.clip(int(abs(forces[2] * scale_factor)), 3, 50)
                # Draw circle
                cv2.circle(img, (center_x + disp_x, center_y + disp_y), radius, (0, 255, 0), -1)

        # Display the image
        cv2.imshow("Sensor Data Visualization", img)
        cv2.waitKey(1)

    def get_forces(self):
        if self.lastmessage["message"] != "No message":
            # print('Got Sensor Reading!')
            self.sensor_reading = self.message_parser(self.lastmessage.copy())
            # self.visualize_sensor_data(self.sensor_reading[1])
            return self.sensor_reading
        else:
            print('[WARNING] No tactile sensor reading detected.')
            return None

# Example usage:
if __name__ == "__main__":
    ip = "131.181.33.191"  # your computer IP on the network
    port = 5000  # the port the server is running on
    sensor_socket = SensorSocket(ip, port)

    while True:
        print(sensor_socket.get_forces())

        