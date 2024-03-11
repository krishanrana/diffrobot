#!/usr/bin/env python3
import websocket
import json
from time import sleep
import threading
import pdb
import numpy as np
import cv2

ip = "131.181.33.191" #your computer IP on the network
port = 5000 #the port the server is running on

lastmessage = {"message":"No message"} #default message you will overwrite when you get update
def on_message(wsapp, message):
    global lastmessage #globalise to overwrite the original
    try:
        data = json.loads(message)
    except Exception:
        pass
    else:
        try:
            if data["message"] == "Welcome":#get the Welcome Message with details, print if you like
                print(data)
            else:
                lastmessage = data
        except Exception:
            pass #ignore message as it's probably invalid


def threader(target, args=False, **targs):
    #args is a tuple of arguments for a threaded function; other key-value pairs will be sent to Thread
    if args:
        targs["args"]=(args,)
    thr = threading.Thread(target=target, **targs)
    thr.daemon = True
    thr.start()


# Function to draw a circle with specified parameters
def draw_circle(image, center, radius, color, thickness=-1):
    cv2.circle(image, center, radius, color, thickness)

# Function to visualize sensor data
def visualize_sensor_data(sensor_data):
    # Create a blank image
    image_size = 400
    img = np.zeros((image_size, image_size, 3), np.uint8)

    # Define grid parameters
    grid_size = 4
    cell_size = image_size // grid_size

    # Loop through each taxel data
    for idx, data in enumerate(sensor_data):
        # Calculate grid position
        row = idx // grid_size
        col = idx % grid_size

        # Extract force components
        x, y, z = data['x'], data['y'], data['z']

        # Calculate circle center position
        center_x = col * cell_size + cell_size // 2
        center_y = row * cell_size + cell_size // 2

        disp_x = int(x * 10)  # Scale for visualization
        disp_y = int(y * 10)  # Scale for visualization

        # Calculate circle radius based on z force
        
        radius = np.clip(int(abs(z * 10)), 3, 50)  # Scale for visualization

        # Draw circle
        draw_circle(img, (center_x + disp_x, center_y + disp_y), radius, (0, 255, 0), -1)

    # Display the image
    cv2.imshow("Sensor Data Visualization", img)
    cv2.waitKey(1)
    # cv2.destroyAllWindows()


def message_parser(msg_obj):
    """
    Parse incoming sensor data messages.

    Args:
        msg_obj (dict): The incoming message object.

    Returns:
        dict: A dictionary containing parsed sensor data, or None if parsing fails.
    """

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
            # Extract force components
            x, y, z = forces[i:i + 3]

            # Filter forces
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



def get_forces():
    # call this function in your policy loop to get the most recent tactile sensor readings
    pass


def mesreader():#this is your app reading the last valid message you received
    while True:#to run forever
        if lastmessage["message"]!="No message":
            # print("I received: {}\n--------".format(str(lastmessage)))
            sensor = message_parser(lastmessage.copy())
            print(sensor['1']['Forces'][0]['x'])
            visualize_sensor_data(sensor['1']['Forces'])
    try: #try to close the app once you press CTRL + C
        wsapp.close()
    except Exception:
        exit()

threader(mesreader,name="Receiver") #start you main app
websocket.setdefaulttimeout(1) #you should avoid increasing it.
wsapp = websocket.WebSocketApp("ws://{}:{}".format(ip,port), on_message=on_message)#set up WebSockets
wsapp.run_forever() #Run until the connection dies
exit()