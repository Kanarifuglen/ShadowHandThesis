import socket
import threading
import time
import os
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.handAngleData import HandAnglesData
import json
import pyrr.quaternion as pyq
import math

# Hvis du vil kjøre denne så må du bytte i rokoko.
# Constants for the socket between rokoko and the converter
IP_SERVER = "172.18.130.99"
PORT_SERVER = 12121
BUFFER_SIZE = 16384

# Constants for the socket between the converter and the hand
SERVER_ADRESS_PORT   = ("172.18.130.99", 20001)

READING_SOCKET_DELAY = 0.1 # delay in seconds to read the socket


# Function to extract the interest data from the rokoko message
def FilterReceiveMessage(message):

    rokokoData= json.loads(message)

    scene = rokokoData['scene']
    actors = scene['actors']
    body = actors[0]['body']

    return {
        'rightThumbProximal': body['rightThumbProximal']['rotation'],
        'rightThumbMedial': body['rightThumbMedial']['rotation'],
        'rightThumbDistal':  body['rightThumbDistal']['rotation'],
        'rightThumbTip': body['rightThumbTip']['rotation'],
        'rightIndexProximal': body['rightIndexProximal']['rotation'],
        'rightIndexMedial': body['rightIndexMedial']['rotation'],
        'rightIndexDistal': body['rightIndexDistal']['rotation'],
        'rightIndexTip': body['rightIndexTip']['rotation'],
        'rightMiddleProximal': body['rightMiddleProximal']['rotation'],
        'rightMiddleMedial': body['rightMiddleMedial']['rotation'],
        'rightMiddleDistal': body['rightMiddleDistal']['rotation'],
        'rightMiddleTip': body['rightMiddleTip']['rotation'],
        'rightRingProximal': body['rightRingProximal']['rotation'],
        'rightRingMedial': body['rightRingMedial']['rotation'],
        'rightRingDistal': body['rightRingDistal']['rotation'],
        'rightRingTip':  body['rightRingTip']['rotation'],
        'rightLittleProximal': body['rightLittleProximal']['rotation'],
        'rightLittleMedial': body['rightLittleMedial']['rotation'],
        'rightLittleDistal': body['rightLittleDistal']['rotation'],
        'rightLittleTip': body['rightLittleTip']['rotation'],
        'rightHand': body['rightHand']['rotation']
    }


# Function to receive the calibration message
# The calibration message is the first message sent by rokoko
# It contains the initial position of the hand

def calibration():
    bytesAddressPair = UDPServerSocket.recvfrom(BUFFER_SIZE)
    print("Calibration message received")
    message = bytesAddressPair[0]
    return FilterReceiveMessage(message)


# Function to convert a quaternion to an angle in degrees
# Return only the angle around the x axis
def computeEulerAngle(quaternion):

    x = quaternion[0]
    y = quaternion[1]
    z = quaternion[2]
    w = quaternion[3]

    phi = math.atan2(2*(w*x + y*z),1-2*(x**2 + y**2))
    phi = - math.degrees(phi)

    return phi


# Function to convert a quaternion to an angle in degrees
# Return the angles around the x and z axis
def computeEulerAngle2(quaternion):

    x = quaternion[0]
    y = quaternion[1]
    z = quaternion[2]
    w = quaternion[3]

    # angles around the x axis
    phi = math.atan2(2*(w*x + y*z),1-2*(x**2 + y**2))
    phi = - math.degrees(phi)

    # angles around the z axis
    psi = math.atan2(2*(w*z + x*y),1-2*(y**2 + z**2))
    psi = math.degrees(psi)

    return phi,psi

# Function to convert a quaternion to an angle in degrees
# Return the angles around the x, y and z axis
def computeEulerAngle3(quaternion):

    x = quaternion[0]
    y = quaternion[1]
    z = quaternion[2]
    w = quaternion[3]

    # angles around the x axis
    phi = math.atan2(2*(w*x + y*z),1-2*(x**2 + y**2))
    phi = - math.degrees(phi)


    # angles around the y axis
    theta = (- math.pi/2) + math.atan2(math.sqrt(1+2*(w*y- x*z)),math.sqrt(1-2*(w*y - x*z)))
    theta = math.degrees(theta)

    # angles around the z axis
    psi = math.atan2(2*(w*z + x*y),1-2*(y**2 + z**2))
    psi = math.degrees(psi)

    return phi,psi,theta


# Function to solve the equation which allows to find the angles from the quaternion 
#  The Details of the resolution are in the documentation.
#The equation is q_z_theta1 * qx_90 * q_z_theta2 = quaternion 
def solveFinalEquation(quaternion) :
    
        W = quaternion[3]
        X = quaternion[0]
        Y = quaternion[1]
        Z = quaternion[2]
    
        A = math.cos(math.pi/4)
    
        theta1plus2 = math.atan2(Z/A,W/A)*2
        theta1moins2 = math.atan2(Y/A,X/A)*2
    
        theta1 = - math.degrees((theta1plus2 + theta1moins2)/2)
        theta2 =  - math.degrees((theta1plus2 - theta1moins2)/2)
    
        return theta1,theta2


# compute the angle of the thumb from the quaternion with the DHKK method
def computeThumbAngleDHKK(q_sensor_p_prime, q_sensor_h_prime) :


    q_h_0 = pyq.create_from_x_rotation(math.radians(45))

    q_2_p = pyq.cross(pyq.create_from_x_rotation(math.radians(90)),pyq.create_from_y_rotation(-math.radians(45)))



    q_0_2 = pyq.cross(pyq.cross(pyq.cross(pyq.inverse(q_h_0),pyq.inverse(q_sensor_h_prime)),q_sensor_p_prime),pyq.inverse(q_2_p))


    # Solving the equation
    theta4,theta5 = solveFinalEquation(q_0_2)

    return theta4,theta5
# create a dictionary of quaternions from the rokoko data with pyrr library
def createQuaternion(rokokoData):
    
        quaternion = {}
    
        for jointName in rokokoData:
            quaternion[jointName] = pyq.create(rokokoData[jointName]['x'],rokokoData[jointName]['y'],rokokoData[jointName]['z'],rokokoData[jointName]['w'])
    
        return quaternion






# Transform the quaternions in relative quaternions in relation to the previous joint and initial position
def transformInRelativeQuaternion(quaternion,qRef):

    relativeQuaternion = {}
        
    relativeQuaternion['q_WRJ1'] = pyq.cross(pyq.inverse(qRef), quaternion['rightHand'])
    relativeQuaternion['q_FFJ1'] = pyq.cross(pyq.inverse(quaternion['rightIndexMedial']), quaternion['rightIndexDistal'])
    relativeQuaternion['q_FFJ2'] = pyq.cross(pyq.inverse(quaternion['rightIndexProximal']), quaternion['rightIndexMedial'])
    relativeQuaternion['q_FFJ3'] = pyq.cross(pyq.inverse(quaternion['rightHand']), quaternion['rightIndexProximal'])

    relativeQuaternion['q_MFJ1'] = pyq.cross(pyq.inverse(quaternion['rightMiddleMedial']), quaternion['rightMiddleDistal'])
    relativeQuaternion['q_MFJ2'] = pyq.cross(pyq.inverse(quaternion['rightMiddleProximal']), quaternion['rightMiddleMedial'])
    relativeQuaternion['q_MFJ3'] = pyq.cross(pyq.inverse(quaternion['rightHand']), quaternion['rightMiddleProximal'])

    relativeQuaternion['q_RFJ1'] = pyq.cross(pyq.inverse(quaternion['rightRingMedial']), quaternion['rightRingDistal'])
    relativeQuaternion['q_RFJ2'] = pyq.cross(pyq.inverse(quaternion['rightRingProximal']), quaternion['rightRingMedial'])
    relativeQuaternion['q_RFJ3'] = pyq.cross(pyq.inverse(quaternion['rightHand']), quaternion['rightRingProximal'])

    relativeQuaternion['q_LFJ1'] = pyq.cross(pyq.inverse(quaternion['rightLittleMedial']), quaternion['rightLittleDistal'])
    relativeQuaternion['q_LFJ2'] = pyq.cross(pyq.inverse(quaternion['rightLittleProximal']), quaternion['rightLittleMedial'])
    relativeQuaternion['q_LFJ3'] = pyq.cross(pyq.inverse(quaternion['rightHand']), quaternion['rightLittleProximal'])
    
    relativeQuaternion['q_THJ1'] = pyq.cross(pyq.inverse(quaternion['rightThumbMedial']), quaternion['rightThumbDistal'])
    relativeQuaternion['q_THJ2'] = pyq.cross(pyq.inverse(quaternion['rightThumbProximal']), quaternion['rightThumbMedial'])
    relativeQuaternion['q_THJ4'] = pyq.cross(pyq.inverse(quaternion['rightHand']), quaternion['rightThumbProximal'])


    return relativeQuaternion


def calculateAnglesFromQuaternion(relativeQuaternion,quaternions):
    handAngle = {}

    # Calcul with Euler Method


    for jointName in relativeQuaternion:

        # The angles are calculated differently for the third quaternion of the fingers. 
        # This quaternion is used to calculate the angles around the x and z axis
        # this corresponds to the angle 3 and 4 of the fingers for the hand
        if jointName == 'q_FFJ3' or jointName == 'q_MFJ3' or jointName == 'q_RFJ3' or jointName == 'q_LFJ3' :
            handAngle['rh'+ jointName[1:]] ,handAngle['rh'+ jointName[1:5]+'4'] = computeEulerAngle2(relativeQuaternion[jointName])


        # The angles are calculated differently for the first quaternion of the wrist
        # This quaternion is used to calculate the angles around the x and z axis
        # this corresponds to the angle 1 and 2 of the wrist for the hand 
        elif jointName == 'q_WRJ1':
            handAngle['rh'+ jointName[1:]] ,handAngle['rh'+ jointName[1:5]+'2']= computeEulerAngle2(relativeQuaternion[jointName])


        elif jointName == 'q_THJ2':
            handAngle['rh'+ jointName[1:]] ,handAngle['rh'+ jointName[1:5]+'3'] = computeEulerAngle2(relativeQuaternion[jointName])

        elif jointName == 'q_THJ4':
            handAngle['rh'+ jointName[1:]] ,_,handAngle['rh'+ jointName[1:5]+'5'] = computeEulerAngle3(relativeQuaternion[jointName])

        else:
            handAngle['rh'+ jointName[1:]]= computeEulerAngle(relativeQuaternion[jointName])

    
    handAngle['rh_THJ4'],handAngle['rh_THJ5'] = computeThumbAngleDHKK(quaternions['rightThumbProximal'],quaternions['rightHand'])
    
    # The angles are adjusted to the hand
    handAngle['rh_FFJ4'] = -handAngle['rh_FFJ4']
    handAngle['rh_MFJ4'] = -handAngle['rh_MFJ4']

    # This angles are not used in this version
    # we leave open the possibility of using them at a later date
    handAngle['rh_LFJ5'] = 0


    return HandAnglesData.fromDict(handAngle)
    
# Function to convert the rokoko data to angles data
def convertFromRokokoDataToAnglesData(rokokoData,initialMessage):

    # Create initiate quaternions from the rokoko data 
    quaternions = createQuaternion(rokokoData)


    qRefRightHand = pyq.create(initialMessage['rightHand']['x'],initialMessage['rightHand']['y'],initialMessage['rightHand']['z'],initialMessage['rightHand']['w'])

    # Transform the quaternions in relative quaternions in relation to the previous joint and initial position
    relativeQuaternion = transformInRelativeQuaternion(quaternions,qRefRightHand)


    # Calculate the angles from the relative quaternions
    return calculateAnglesFromQuaternion(relativeQuaternion,quaternions)


# Function to send the angles to the hand called by the thread
def sendAngles(queue,initialMessage,UDPClientSocket):


    while(True):

        if  len(queue) != 0 :
        
            # Get the angles from the queue
            data = queue.pop()

            # Convert the quaternion to angles readable by the hand  
            thetas = convertFromRokokoDataToAnglesData(data,initialMessage)
            thetas.convertToInt()

            # send the angles to the hand
            data_to_send = thetas.to_struct()

            UDPClientSocket.sendto(data_to_send, SERVER_ADRESS_PORT)

            # # # # # Wait an answer from the hand
            # UDPClientSocket.recvfrom(BUFFER_SIZE)
        
            # Clear the queue
            queue.clear()

        time.sleep(READING_SOCKET_DELAY)





# Create a datagram socket to receive the angles from rokoko
UDPServerSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

# Bind to address and ip
UDPServerSocket.bind((IP_SERVER, PORT_SERVER))
print("UDP server up and listening")


# Create a UDP socket at client to send the angles to the hand

UDPClientSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

# Get the initial position of the hand
initialMessage = calibration()


# Create a queue to store the angles
queue = []


# Create a thread to send the angles to the hand
t1 = threading.Thread(target=sendAngles, args=(queue,initialMessage,UDPClientSocket))
t1.start()

while(True):

    # Read socket
    bytesAddressPair = UDPServerSocket.recvfrom(BUFFER_SIZE)
    message = bytesAddressPair[0]
    address = bytesAddressPair[1]

    # Store the angles in the queue
    queue.append(FilterReceiveMessage(message))

    # Sending a reply to client
    msgFromServer = "Hello UDP Client"
    bytesToSend = str.encode(msgFromServer)
    UDPServerSocket.sendto(bytesToSend, address)