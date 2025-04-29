import numpy as np
import struct

# Class to store the angles of the hand 
# WR signifying wrist, FF the four fingers, MF the middle finger, RF the ring finger, LF the little finger, TH the thumb
# joints are ordered from the tip to the base
class ArmAnglesData:
    def __init__(self, rh_SHJ1, rh_SHJ2, rh_ELJ1,rh_WRJ1,rh_WRJ2,rh_WRJ3):
        self.rh_SHJ1 = rh_SHJ1
        self.rh_SHJ2 = rh_SHJ2
        self.rh_ELJ1 = rh_ELJ1
        self.rh_WRJ1 = rh_WRJ1
        self.rh_WRJ2 = rh_WRJ2
        self.rh_WRJ3 = rh_WRJ3
    

    # Convert the angles to a dictionary
    @classmethod
    def fromDict(self, data):
        return ArmAnglesData(data['rh_SHJ1'], data['rh_SHJ2'], data['rh_ELJ1'],data['rh_WRJ1'],data['rh_WRJ2'],data['rh_WRJ3'])
    
    @classmethod
    def fromStruct(self,data):
        rh_SHJ1, rh_SHJ2, rh_ELJ1,rh_WRJ1,rh_WRJ2,rh_WRJ3 = struct.unpack('!iiiiii', data)
        return ArmAnglesData(rh_SHJ1, rh_SHJ2, rh_ELJ1,rh_WRJ1,rh_WRJ2,rh_WRJ3)


    # Convert the angles to a string
    def __str__(self):
        return f'rh_SHJ1: {self.rh_SHJ1}, \n\
        rh_SHJ2: {self.rh_SHJ2}, \n\
        rh_ELJ1: {self.rh_ELJ1}, \n\
        rh_WRJ1: {self.rh_WRJ1}, \n\
        rh_WRJ2: {self.rh_WRJ2}, \n\
        rh_WRJ3: {self.rh_WRJ3}'

    
    # Convert the angles to an array
    def to_array(self) :
        return np.array([self.rh_SHJ1, self.rh_SHJ2, self.rh_ELJ1,self.rh_WRJ1,self.rh_WRJ2,self.rh_WRJ3])
    
    
    # Convert the angles to a struct
    def to_struct(self):
        return struct.pack('!iiiiii',
                          self.rh_SHJ1, self.rh_SHJ2, self.rh_ELJ1,self.rh_WRJ1,self.rh_WRJ2,self.rh_WRJ3)
    

    def convertToInt(self):

        self.rh_WRJ1 = int(self.rh_WRJ1)
        self.rh_WRJ2 = int(self.rh_WRJ2)
        self.rh_WRJ3 = int(self.rh_WRJ3)
        self.rh_SHJ1 = int(self.rh_SHJ1)
        self.rh_SHJ2 = int(self.rh_SHJ2)
        self.rh_ELJ1 = int(self.rh_ELJ1)
 
    # Copy the angles from another object
    def copy(self,angleData) :
        self.rh_WRJ1 = angleData.rh_WRJ1
        self.rh_WRJ2 = angleData.rh_WRJ2
        self.rh_WRJ3 = angleData.rh_WRJ3
        self.rh_SHJ1 = angleData.rh_SHJ1
        self.rh_SHJ2 = angleData.rh_SHJ2
        self.rh_ELJ1 = angleData.rh_ELJ1



        

    

    

