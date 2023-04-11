import cv2 as cv
from math import atan, atan2, cos, sin, sqrt, pi, hypot, asin
import numpy as np
import subprocess
import serial
import time
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
import albumentations.pytorch as AP
from albumentations import (Compose)

class MyMobileV3Net(torch.nn.Module):
    def __init__(self, model_name='mobilenetv3_large_100', pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)

        self.model.classifier = torch.nn.Sequential(
          torch.nn.Linear(1280, 512),
          torch.nn.ReLU(),
          torch.nn.Linear(512, 5),
        )

    def forward(self, x):
        x = self.model(x)
        return x

class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic']

pred_transforms = Compose([
    A.Resize(224, 224),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
    AP.ToTensorV2()])

device = torch.device('cpu')
model = MyMobileV3Net('mobilenetv3_large_100', pretrained=False)

def predict_image(image):
    image_tensor = pred_transforms(image=image)["image"].unsqueeze(0)
    #image_tensor = torch.tensor(image_tensor.clone().detach(), dtype=torch.float)
    image_tensor = image_tensor.clone().detach()
    input = image_tensor.to(device)
    outputs = model(input)
    _, preds = torch.max(outputs, 1)
    prob = F.softmax(outputs, dim=1)
    top_p, top_class = prob.topk(1, dim=1)
    result = class_names[int(preds.cpu().numpy())]
    return result

def serial_port_setup():
    ser = serial.Serial(
    port='/dev/ttyUSB0',\
    baudrate=9600,\
    parity=serial.PARITY_NONE,\
    stopbits=serial.STOPBITS_ONE,\
    bytesize=serial.EIGHTBITS,\
    timeout=None)

    return ser

def decrease_brightness(img, value):
    image = cv.imread(img)
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    hsv[:,:,2] -= value
    image = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    cv.imwrite(img, image)

def increase_contrast(img):
    image = cv.imread(img)
    lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv.merge((cl,a,b))
    final = cv.cvtColor(limg, cv.COLOR_LAB2BGR)
    cv.imwrite(img, final)

def get_center_coords(image):
    # Load the image
    img = cv.imread(image)

    # Was the image there?
    if img is None:
        print("Error: File not found")
        exit(0)

    # Convert image to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Convert image to binary
    _, bw = cv.threshold(gray, 50, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

    # Find all the contours in the thresholded image
    contours, _ = cv.findContours(bw, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

    for i, c in enumerate(contours):

        # Calculate the area of each contour
        area = cv.contourArea(c)

        # Ignore contours that are too small or too large
        #if area < 3700 or 100000 < area:
        if area < 9000 or 100000 < area:
            continue

        rect = cv.minAreaRect(c)
        box = cv.boxPoints(rect)
        box = np.int0(box)

        cv.drawContours(img,[box],0,(0,0,255),2)
        cv.imwrite("new.jpg", img)

        # Retrieve the key parameters of the rotated bounding box
        center = (int(rect[0][0]),int(rect[0][1]))
        width = int(rect[1][0])
        height = int(rect[1][1])
        angle = int(rect[2])

    return int(rect[0][0]), int(rect[0][1])

def get_rotate_angle(x, y):

    img_height = 1380
    img_width = 1280
    x_robot = 708

    y_object = img_height -y

    if x < x_robot:
      x_object = x_robot - x
      angle = int(np.degrees(atan(y_object / x_object)))
    else:
      x_object = x - x_robot
      angle = 90 - int(np.degrees(atan(y_object / x_object))) + 66

    return angle

def get_tilt_angle(dist):

    h = 639
    a = 733
    if (dist > a):
        b = dist - a
        c = hypot(h, b)
        angle = 175 - int(np.degrees(asin(b / c))) - 5

    else:
        b = a - dist
        c = hypot(h, b)
        angle = int(np.degrees(asin(b / c)))
        angle = 90 + angle

    return angle

def get_distance(x, y):

    img_height = 1380
    x_robot = 588

    x_object = x - x_robot
    y_object = img_height -y

    distance = int(sqrt(y_object*y_object + x_object*x_object))
    return distance

def main():
    offset = 0
    buff = ""
    serport = serial_port_setup()
    model.load_state_dict(torch.load('model/mobilenetv3_large_100_waste.pth', map_location=torch.device('cpu')))
    model.eval()
    model.to(device)
    print("[INFO] Waiting for event from manipulator...")

    while True:

        while True:
            oneByte = serport.read(1)
            if oneByte == b"\r":
                break
            else:
                buff += oneByte.decode("ascii", "ignore")

        index = buff.find("button event")
        if index >= 0:
            print("[INFO] Button event from manipulator!")
            print("[INFO] Take a photo...")

            error = 1
            while(error):
                subprocess.run("fswebcam --no-banner -d /dev/video0 -r 1920x1080 --jpeg 100 new.jpg", shell=True)
                decrease_brightness("new.jpg", 45)
                #increase_contrast("new.jpg")
                image = cv.imread("new.jpg")
                image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                prediction = predict_image(image)
                image = cv.imread("new.jpg", 0)
                if cv.countNonZero(image) == 0:
                    print("[ERROR] Image is black!")
                else:
                    error = 0

            cv.imshow('Input Image', image)

            x_coord, y_coord = get_center_coords("new.jpg")
            angle = get_rotate_angle(x_coord, y_coord)
            dist = get_distance(x_coord, y_coord)
            print(x_coord, y_coord, angle, dist)

            serport.write((str(angle + offset)).encode())
            fin0A = b'\x0A'
            serport.write(fin0A)
            time.sleep(0.3)

            serport.write(prediction.encode())
            fin0A = b'\x0A'
            serport.write(fin0A)

            buff = ""

            while True:
                oneByte = serport.read(1)
                if oneByte == b"\r":
                    break
                else:
                    buff += oneByte.decode("ascii", "ignore")

            print(buff)

            buff = ""

            print("[INFO] Waiting for event from manipulator...")
            continue


if __name__ == "__main__":
	main()
