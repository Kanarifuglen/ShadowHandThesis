import numpy as np
import struct

# Class to store the angles of the hand 
# WR signifying wrist, FF the four fingers, MF the middle finger, RF the ring finger, LF the little finger, TH the thumb
# joints are ordered from the tip to the base
class HandAnglesData:
    def __init__(self, rh_WRJ1, rh_WRJ2, rh_FFJ4, rh_FFJ3, rh_FFJ2, rh_FFJ1, rh_MFJ4, rh_MFJ3, rh_MFJ2, rh_MFJ1, rh_RFJ4, rh_RFJ3, rh_RFJ2, rh_RFJ1, rh_LFJ4, rh_LFJ3, rh_LFJ2, rh_LFJ1, rh_THJ4, rh_THJ3, rh_THJ2, rh_THJ1, rh_THJ5, rh_LFJ5):
        self.rh_WRJ1 = rh_WRJ1
        self.rh_WRJ2 = rh_WRJ2
        self.rh_FFJ4 = rh_FFJ4
        self.rh_FFJ3 = rh_FFJ3
        self.rh_FFJ2 = rh_FFJ2
        self.rh_FFJ1 = rh_FFJ1
        self.rh_MFJ4 = rh_MFJ4
        self.rh_MFJ3 = rh_MFJ3
        self.rh_MFJ2 = rh_MFJ2
        self.rh_MFJ1 = rh_MFJ1
        self.rh_RFJ4 = rh_RFJ4
        self.rh_RFJ3 = rh_RFJ3
        self.rh_RFJ2 = rh_RFJ2
        self.rh_RFJ1 = rh_RFJ1
        self.rh_LFJ4 = rh_LFJ4
        self.rh_LFJ3 = rh_LFJ3
        self.rh_LFJ2 = rh_LFJ2
        self.rh_LFJ1 = rh_LFJ1
        self.rh_THJ4 = rh_THJ4
        self.rh_THJ3 = rh_THJ3
        self.rh_THJ2 = rh_THJ2
        self.rh_THJ1 = rh_THJ1
        self.rh_THJ5 = rh_THJ5
        self.rh_LFJ5 = rh_LFJ5
    

    @classmethod
    def fromStruct(self,data):
        rh_WRJ1, rh_WRJ2, rh_FFJ4, rh_FFJ3, rh_FFJ2, rh_FFJ1, rh_MFJ4, rh_MFJ3, rh_MFJ2, rh_MFJ1, rh_RFJ4, rh_RFJ3, rh_RFJ2, rh_RFJ1, rh_LFJ4, rh_LFJ3, rh_LFJ2, rh_LFJ1, rh_THJ4, rh_THJ3, rh_THJ2, rh_THJ1, rh_THJ5,rh_LFJ5  = struct.unpack('!iiiiiiiiiiiiiiiiiiiiiiii', data)
        return HandAnglesData(rh_WRJ1, rh_WRJ2, rh_FFJ4, rh_FFJ3, rh_FFJ2, rh_FFJ1, rh_MFJ4, rh_MFJ3, rh_MFJ2, rh_MFJ1, rh_RFJ4, rh_RFJ3, rh_RFJ2, rh_RFJ1, rh_LFJ4, rh_LFJ3, rh_LFJ2, rh_LFJ1, rh_THJ4, rh_THJ3, rh_THJ2, rh_THJ1, rh_THJ5, rh_LFJ5)


    # Convert the angles to a dictionary
    @classmethod
    def fromDict(self, data):
        return HandAnglesData(data['rh_WRJ1'], data['rh_WRJ2'], data['rh_FFJ4'], data['rh_FFJ3'], data['rh_FFJ2'], data['rh_FFJ1'], data['rh_MFJ4'], data['rh_MFJ3'], data['rh_MFJ2'], data['rh_MFJ1'], data['rh_RFJ4'], data['rh_RFJ3'], data['rh_RFJ2'], data['rh_RFJ1'], data['rh_LFJ4'], data['rh_LFJ3'], data['rh_LFJ2'], data['rh_LFJ1'], data['rh_THJ4'], data['rh_THJ3'], data['rh_THJ2'], data['rh_THJ1'], data['rh_THJ5'], data['rh_LFJ5'])


    # Convert the angles to a string
    def __str__(self):
        return f' rh_WRJ1: {self.rh_WRJ1}, \n\
        rh_WRJ2: {self.rh_WRJ2},\n\
        rh_THJ1: {self.rh_THJ1}, \n\
        rh_THJ2: {self.rh_THJ2}, \n\
        rh_THJ3: {self.rh_THJ3}, \n\
        rh_THJ4: {self.rh_THJ4}, \n\
        rh_THJ5: {self.rh_THJ5}, \n\
        rh_WRJ1: {self.rh_WRJ1}, \n\
        rh_WRJ2: {self.rh_WRJ2}, \n\
        rh_FFJ4: {self.rh_FFJ4}, \n\
        rh_FFJ3: {self.rh_FFJ3}, \n\
        rh_FFJ2: {self.rh_FFJ2}, \n\
        rh_FFJ1: {self.rh_FFJ1}, \n\
        rh_MFJ4: {self.rh_MFJ4}, \n\
        rh_MFJ3: {self.rh_MFJ3}, \n\
        rh_MFJ2: {self.rh_MFJ2}, \n\
        rh_MFJ1: {self.rh_MFJ1}, \n\
        rh_RFJ4: {self.rh_RFJ4}, \n\
        rh_RFJ3: {self.rh_RFJ3}, \n\
        rh_RFJ2: {self.rh_RFJ2}, \n\
        rh_RFJ1: {self.rh_RFJ1}, \n\
        rh_LFJ4: {self.rh_LFJ4}, \n\
        rh_LFJ3: {self.rh_LFJ3}, \n\
        rh_LFJ2: {self.rh_LFJ2}, \n\
        rh_LFJ1: {self.rh_LFJ1}, \n\
        rh_THJ1: {self.rh_THJ1}, \n\
        rh_THJ2: {self.rh_THJ2}, \n\
        rh_THJ3: {self.rh_THJ3}, \n\
        rh_THJ4: {self.rh_THJ4}, \n\
        rh_THJ5: {self.rh_THJ5}, \n\
        rh_LFJ5: {self.rh_LFJ5}'

    
    
    # Convert the angles to an array
    def to_array(self) :
        return np.array([[self.rh_WRJ1, self.rh_WRJ2,
                         self.rh_FFJ4, self.rh_FFJ3, self.rh_FFJ2, self.rh_FFJ1,
                         self.rh_MFJ4, self.rh_MFJ3, self.rh_MFJ2, self.rh_MFJ1,
                         self.rh_RFJ4, self.rh_RFJ3, self.rh_RFJ2, self.rh_RFJ1, 
                         self.rh_LFJ5, self.rh_LFJ4, self.rh_LFJ3, self.rh_LFJ2, self.rh_LFJ1,
                         self.rh_THJ5, self.rh_THJ4, self.rh_THJ3, self.rh_THJ2, self.rh_THJ1]])
    
    
    # Convert the angles to a struct
    def to_struct(self):
        return struct.pack('!iiiiiiiiiiiiiiiiiiiiiiii',
                           self.rh_WRJ1, self.rh_WRJ2,
                           self.rh_FFJ4, self.rh_FFJ3, self.rh_FFJ2, self.rh_FFJ1,
                           self.rh_MFJ4, self.rh_MFJ3, self.rh_MFJ2, self.rh_MFJ1,
                           self.rh_RFJ4, self.rh_RFJ3, self.rh_RFJ2, self.rh_RFJ1,
                           self.rh_LFJ4, self.rh_LFJ3, self.rh_LFJ2, self.rh_LFJ1,
                           self.rh_THJ4, self.rh_THJ3, self.rh_THJ2, self.rh_THJ1,self.rh_THJ5,self.rh_LFJ5)
    

    def convertToInt(self):

        self.rh_WRJ1 = int(self.rh_WRJ1)
        self.rh_WRJ2 = int(self.rh_WRJ2)
        self.rh_FFJ4 = int(self.rh_FFJ4)
        self.rh_FFJ3 = int(self.rh_FFJ3)
        self.rh_FFJ2 = int(self.rh_FFJ2)
        self.rh_FFJ1 = int(self.rh_FFJ1)
        self.rh_MFJ4 = int(self.rh_MFJ4)
        self.rh_MFJ3 = int(self.rh_MFJ3)
        self.rh_MFJ2 = int(self.rh_MFJ2)
        self.rh_MFJ1 = int(self.rh_MFJ1)
        self.rh_RFJ4 = int(self.rh_RFJ4)
        self.rh_RFJ3 = int(self.rh_RFJ3)
        self.rh_RFJ2 = int(self.rh_RFJ2)
        self.rh_RFJ1 = int(self.rh_RFJ1)
        self.rh_LFJ4 = int(self.rh_LFJ4)
        self.rh_LFJ3 = int(self.rh_LFJ3)
        self.rh_LFJ2 = int(self.rh_LFJ2)
        self.rh_LFJ1 = int(self.rh_LFJ1)
        self.rh_THJ4 = int(self.rh_THJ4)
        self.rh_THJ3 = int(self.rh_THJ3)
        self.rh_THJ2 = int(self.rh_THJ2)
        self.rh_THJ1 = int(self.rh_THJ1)
        self.rh_THJ5 = int(self.rh_THJ5)
        self.rh_LFJ5 = int(self.rh_LFJ5)


        
    # limit angles to the hand's range of motion
    def limitAngles(self):  

        if (self.rh_FFJ1 < 0) :
            self.rh_FFJ1 = 0
        if (self.rh_FFJ1 > 90) :
            self.rh_FFJ1 = 90
        
        if (self.rh_FFJ2 < 0) :
            self.rh_FFJ2 = 0
        if (self.rh_FFJ2 > 90) :
            self.rh_FFJ2 = 90

        if (self.rh_FFJ3 < -15) :
            self.rh_FFJ3 = -15
        if (self.rh_FFJ3 > 90) :
            self.rh_FFJ3 = 90
        
        if (self.rh_FFJ4 < -20) :
            self.rh_FFJ4 = -20
        if (self.rh_FFJ4 > 20) :
            self.rh_FFJ4 = 20

        if (self.rh_MFJ1 < 0) :
            self.rh_MFJ1 = 0
        if (self.rh_MFJ1 > 90) :
            self.rh_MFJ1 = 90

        if (self.rh_MFJ2 < 0) :
            self.rh_MFJ2 = 0
        if (self.rh_MFJ2 > 90) :
            self.rh_MFJ2 = 90
        
        if (self.rh_MFJ3 < -15) :
            self.rh_MFJ3 = -15
        if (self.rh_MFJ3 > 90) :
            self.rh_MFJ3 = 90

        if (self.rh_MFJ4 < -20) :
            self.rh_MFJ4 = -20
        if (self.rh_MFJ4 > 20) :
            self.rh_MFJ4 = 20
        
        if (self.rh_RFJ1 < 0) :
            self.rh_RFJ1 = 0
        if (self.rh_RFJ1 > 90) :
            self.rh_RFJ1 = 90

        if (self.rh_RFJ2 < 0) :
            self.rh_RFJ2 = 0
        if (self.rh_RFJ2 > 90) :
            self.rh_RFJ2 = 90

        if (self.rh_RFJ3 < -15) :
            self.rh_RFJ3 = -15
        if (self.rh_RFJ3 > 90) :
            self.rh_RFJ3 = 90

        if (self.rh_RFJ4 < -20) :
            self.rh_RFJ4 = -20
        if (self.rh_RFJ4 > 20) :
            self.rh_RFJ4 = 20

        if (self.rh_LFJ1 < 0) :
            self.rh_LFJ1 = 0
        if (self.rh_LFJ1 > 90) :
            self.rh_LFJ1 = 90

        if (self.rh_LFJ2 < 0) :
            self.rh_LFJ2 = 0
        if (self.rh_LFJ2 > 90) :
            self.rh_LFJ2 = 90

        if (self.rh_LFJ3 < -15) :
            self.rh_LFJ3 = -15
        if (self.rh_LFJ3 > 90) :
            self.rh_LFJ3 = 90

        if (self.rh_LFJ4 < -20) :
            self.rh_LFJ4 = -20
        if (self.rh_LFJ4 > 20) :
            self.rh_LFJ4 = 20

        if(self.rh_LFJ5 < 0):
            self.rh_LFJ5 = 0
        if(self.rh_LFJ5 > 45):
            self.rh_LFJ5 = 45

        if (self.rh_THJ1 < -15) :
            self.rh_THJ1 = -15
        if (self.rh_THJ1 > 90) :
            self.rh_THJ1 = 90
        
        if (self.rh_THJ2 < -40) :
            self.rh_THJ2 = -40
        if (self.rh_THJ2 > 40) :
            self.rh_THJ2 = 40
        
        if (self.rh_THJ3 < -12) :
            self.rh_THJ3 = -12
        if (self.rh_THJ3 > 12) :
            self.rh_THJ3 = 12
        
        if (self.rh_THJ4 < 0) :
            self.rh_THJ4 = 0
        if (self.rh_THJ4 > 70) :
            self.rh_THJ4 = 70
        
        if (self.rh_THJ5 < -60) :
            self.rh_THJ5 = -60
        if (self.rh_THJ5 > 60) :
            self.rh_THJ5 = 60

        if (self.rh_WRJ1 < -40) :
            self.rh_WRJ1 = -40
        if (self.rh_WRJ1 > 28) :
            self.rh_WRJ1 = 28

        if (self.rh_WRJ2 < -28) :
            self.rh_WRJ2 = -28
        if (self.rh_WRJ2 > 10) :
            self.rh_WRJ2 = 10
  

    def copy(self,angleData) :
        self.rh_WRJ1 = angleData.rh_WRJ1
        self.rh_WRJ2 = angleData.rh_WRJ2
        self.rh_FFJ4 = angleData.rh_FFJ4
        self.rh_FFJ3 = angleData.rh_FFJ3
        self.rh_FFJ2 = angleData.rh_FFJ2
        self.rh_FFJ1 = angleData.rh_FFJ1
        self.rh_MFJ4 = angleData.rh_MFJ4
        self.rh_MFJ3 = angleData.rh_MFJ3
        self.rh_MFJ2 = angleData.rh_MFJ2
        self.rh_MFJ1 = angleData.rh_MFJ1
        self.rh_RFJ4 = angleData.rh_RFJ4
        self.rh_RFJ3 = angleData.rh_RFJ3
        self.rh_RFJ2 = angleData.rh_RFJ2
        self.rh_RFJ1 = angleData.rh_RFJ1
        self.rh_LFJ4 = angleData.rh_LFJ4
        self.rh_LFJ3 = angleData.rh_LFJ3
        self.rh_LFJ2 = angleData.rh_LFJ2
        self.rh_LFJ1 = angleData.rh_LFJ1
        self.rh_THJ4 = angleData.rh_THJ4
        self.rh_THJ3 = angleData.rh_THJ3
        self.rh_THJ2 = angleData.rh_THJ2
        self.rh_THJ1 = angleData.rh_THJ1
        self.rh_THJ5 = angleData.rh_THJ5
        self.rh_LFJ5 = angleData.rh_LFJ5

    def copyFirst(self,angleData) :
        self.rh_FFJ4 = angleData.rh_FFJ4
        self.rh_FFJ3 = angleData.rh_FFJ3
        self.rh_FFJ2 = angleData.rh_FFJ2
        self.rh_FFJ1 = angleData.rh_FFJ1

    def copyMiddle(self,angleData) :
        self.rh_MFJ4 = angleData.rh_MFJ4
        self.rh_MFJ3 = angleData.rh_MFJ3
        self.rh_MFJ2 = angleData.rh_MFJ2
        self.rh_MFJ1 = angleData.rh_MFJ1

    def copyRing(self,angleData) :
        self.rh_RFJ4 = angleData.rh_RFJ4
        self.rh_RFJ3 = angleData.rh_RFJ3
        self.rh_RFJ2 = angleData.rh_RFJ2
        self.rh_RFJ1 = angleData.rh_RFJ1

    def copyLittle(self,angleData) :
        self.rh_LFJ5 = angleData.rh_LFJ5
        self.rh_LFJ4 = angleData.rh_LFJ4
        self.rh_LFJ3 = angleData.rh_LFJ3
        self.rh_LFJ2 = angleData.rh_LFJ2
        self.rh_LFJ1 = angleData.rh_LFJ1

    def copyThumb(self,angleData) :
        self.rh_THJ5 = angleData.rh_THJ5
        self.rh_THJ4 = angleData.rh_THJ4
        self.rh_THJ3 = angleData.rh_THJ3
        self.rh_THJ2 = angleData.rh_THJ2
        self.rh_THJ1 = angleData.rh_THJ1


    def noThumb(self):

        self.rh_THJ3 = 0
        self.rh_THJ4 = 0
        self.rh_THJ5 = 0
        self.rh_THJ1 = 0
        self.rh_THJ2 = 0
        

    def noFingers(self):

        self.rh_FFJ4 = 0
        self.rh_FFJ3 = 0
        self.rh_FFJ2 = 0
        self.rh_FFJ1 = 0

        self.rh_MFJ4 = 0
        self.rh_MFJ3 = 0
        self.rh_MFJ2 = 0
        self.rh_MFJ1 = 0

        self.rh_RFJ4 = 0
        self.rh_RFJ3 = 0
        self.rh_RFJ2 = 0
        self.rh_RFJ1 = 0

        self.rh_LFJ5 = 0
        self.rh_LFJ4 = 0
        self.rh_LFJ3 = 0
        self.rh_LFJ2 = 0
        self.rh_LFJ1 = 0


        

    

    

