import websocket
import json
import threading
import numpy as np
import cv2

class TactileSensor:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.lastmessage = {"message":"No message"}
        self.wsapp = websocket.WebSocketApp("ws://{}:{}".format(ip, port), on_message=self.on_message)
        self.threader(self.mesreader, name="Receiver")

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

    def mesreader(self):
        while True:
            if self.lastmessage["message"] != "No message":
                sensor = self.message_parser(self.lastmessage.copy())
                # Call a function to process or visualize sensor data
                self.visualize_sensor_data(sensor['1']['Forces'])
    
    def message_parser(self, msg_obj):
        sensors_data = {}
        for sensor_idx in range(msg_obj['sensors']):
            values = []
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
            for val in data_val:
                try:
                    values.append(int(val, 16))
                except ValueError:
                    print(f"Value `{val}` is not a valid HEX number")
                    return None
            taxels_val = []
            for i in range(0, len(values), 3):
                temp_dict = {}
                temp_dict['x'], temp_dict['y'], temp_dict['z'] = values[i:i + 3]
                taxels_val.append(temp_dict)
            forces_val = []
            for i in range(0, len(forces), 3):
                temp_dict = {}
                x, y, z = forces[i:i + 3]
                temp_dict['x'] = np.clip(x, 0.5, 10) - 0.5
                temp_dict['y'] = np.clip(y, 0.5, 10) - 0.5
                temp_dict['z'] = np.clip(z, 0.5, 10) - 0.5
                forces_val.append(temp_dict)
            sensor_data_dict = {
                'Taxels': taxels_val,
                'Forces': forces_val
            }
            sensors_data[str(sensor_idx + 1)] = sensor_data_dict
        return sensors_data

    def visualize_sensor_data(self, sensor_data):
        image_size = 400
        img = np.zeros((image_size, image_size, 3), np.uint8)
        grid_size = 4
        cell_size = image_size // grid_size
        for idx, data in enumerate(sensor_data):
            row = idx // grid_size
            col = idx % grid_size
            x, y, z = data['x'], data['y'], data['z']
            center_x = col * cell_size + cell_size // 2
            center_y = row * cell_size + cell_size // 2
            disp_x = int(x * 10)  
            disp_y = int(y * 10)  
            radius = np.clip(int(abs(z * 10)), 3, 50)  
            cv2.circle(img, (center_x + disp_x, center_y + disp_y), radius, (0, 255, 0), -1)
        cv2.imshow("Sensor Data Visualization", img)
        cv2.waitKey(1)


    def get_forces(self):
        if self.lastmessage["message"] != "No message":
            sensor = self.message_parser(self.lastmessage.copy())
            # Assuming you want to return forces for sensor 1 as an example
            return sensor.get('1', {}).get('Forces', [])
        else:
            return None

# Example usage:
if __name__ == "__main__":
    ip = "131.181.33.191"  # your computer IP on the network
    port = 5000  # the port the server is running on
    sensor_socket = TactileSensor(ip, port)
    while True:
        print(sensor_socket.get_forces())
