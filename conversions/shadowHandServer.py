"""
This script was developed as part of the Rokoko to Robot pipeline by
Charlotte Andreff and Loïc Blanc, SINLAB, University of Oslo (2024).
Repository: https://github.com/sinlab-uio/rokoko-to-robots

For details, see the documentation in 'Rokoko_to_robot_documentation.pdf'.
"""


import numpy as np
import matplotlib.pyplot as plt
import rospy
from sr_robot_commander.sr_hand_commander import SrHandCommander
import socket
import threading
import time
from tools.handAngleData import HandAnglesData
from display.Graph3DHand import drawHand
from safety.handSafetyModule import safetyMode




READING_SOCKET_DELAY = 0.1 # delay in seconds to read the socket
MOTION_DELAY = 0.1 # delay in seconds between each motion
WAIT = False


# Constants for the socket
IP_SERVER = "172.18.130.99"
PORT_SERVER = 20001
BUFFER_SIZE = 16384




# Function to convert an array to a dictionary 
def arrayToDict(array):
    return {'rh_WRJ1':array[0],'rh_WRJ2':array[1],
            'rh_FFJ4':array[2], 'rh_FFJ3':array[3], 'rh_FFJ2':array[4], 'rh_FFJ1':array[5], 
            'rh_MFJ4':array[6], 'rh_MFJ3':array[7], 'rh_MFJ2':array[8], 'rh_MFJ1':array[9],
            'rh_RFJ4':array[10], 'rh_RFJ3':array[11], 'rh_RFJ2':array[12], 'rh_RFJ1':array[13], 
            'rh_LFJ5':array[14],'rh_LFJ4':array[15], 'rh_LFJ3':array[16], 'rh_LFJ2':array[17], 'rh_LFJ1':array[18], 
            'rh_THJ5':array[19], 'rh_THJ4':array[20], 'rh_THJ3':array[21], 'rh_THJ2':array[22], 'rh_THJ1':array[23]}



# Function to send the angles to the hand called by the thread
def sendAngles(queue):


    # initialize data for kalman filter
    #theta_filtered = np.zeros((48,1))
    #P_kalm_k_= np.eye(48)

    while(rospy.is_shutdown() == False):


        if  len(queue) != 0 :

        
            # Get the angles from the queue
            thetas = queue.pop()


            #safety module
            thetas = safetyMode(thetas)

            # blocks the thumb angles
            thetas.noThumb()
            #thetas.noFingers()

            #Add to the tail of the graph if you want to visualize with the 3D graph
            #queueGraph.append(thetas)


            # Convert the angles to an array
            thetasArray = thetas.to_array().T
            

            # Apply the kalman filter
            #theta_filtered, P_kalm_k_ = kalmanFilter(data.to_array().T,theta_filtered,P_kalm_k_)
            #thetasArray = theta_filtered[::2]  

            # Send the angles to the hand
            joint_states = arrayToDict(thetasArray)
            hand_commander.move_to_joint_value_target_unsafe(joint_states, time = MOTION_DELAY, wait=WAIT, angle_degrees=True)	

            # Clear the queue
            queue.clear()

        time.sleep(READING_SOCKET_DELAY)



# Function to initialize the hand
def initHand():

    # Initialize the ROS node
    rospy.init_node("robot_commander_examples", anonymous=True)
    print("Hand inited")

    # Create a SrHandCommander object
    hand_commander = SrHandCommander(name="right_hand")

    # Print some information about the hand
    print("Robot name: ", hand_commander.get_robot_name())
    print("Group name: ", hand_commander.get_group_name())
    print("Planning frame: ", hand_commander.get_planning_frame())


    # Initialize the hand with the open position
    trajectory = [{'name': 'open', 'interpolate_time': 4.0}] 
    hand_commander.run_named_trajectory(trajectory)
    return hand_commander





# initialize the hand
hand_commander = initHand()
print("Hand initialized")

# Create a datagram socket
UDPServerSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

# Bind to address and ip
UDPServerSocket.bind((IP_SERVER, PORT_SERVER))
print("UDP server up and listening")



# Create a queue to store the angles
queue = []

# Create a thread to draw the hand
#queueGraph = [] 
#t1 = threading.Thread(target=drawHand, args=(queueGraph,"End",))
#t1.start()

# Create a thread to send the angles to the hand
t2 = threading.Thread(target=sendAngles, args=(queue,))
t2.start()
 


while(rospy.is_shutdown() == False):

    # Read socket

    bytesAddressPair = UDPServerSocket.recvfrom(BUFFER_SIZE)
    message = bytesAddressPair[0]
    address = bytesAddressPair[1]


    # Store the angles in the queue
    queue.append(HandAnglesData.fromStruct(message))

    # Sending a reply to client
    msgFromServer = "Hello UDP Client"
    bytesToSend = str.encode(msgFromServer)
    UDPServerSocket.sendto(bytesToSend, address) 






