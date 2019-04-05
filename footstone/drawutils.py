import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import numpy as np

'''
(a) Ax + By + C = 0
(b) (y - y0) / (x - x0) = B / A

=>
x = (B*B*x0 - A*B*y0 - A*C) / (A*A + B*B)
y = (-A*B*x0 + A*A*y0 - B*C) / (A*A + B*B)
'''
def get_footpoint(px, py, w_):
    if w_.shape[0] != 3:
        print("can't calculate footpoint with {}".formate(w_))
        return None
    
    A,B,C = w_[1], w_[2], w_[0]
    x = (B*B*px - A*B*py - A*C)/(A*A + B*B)
    y = (-A*B*px + A*A*py - B*C)/(A*A + B*B)
    
    return x, y

def get_angle(p0, p1=np.array([0,0]), p2=None):
    ''' compute angle (in degrees) for p0p1p2 corner
    Inputs:
        p0,p1,p2 - points in the form of [x,y]
    '''
    if p2 is None:
        p2 = p1 + np.array([1, 0])
    v0 = np.array(p0) - np.array(p1)
    v1 = np.array(p2) - np.array(p1)

    angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
    return np.degrees(angle)

def rotation_transform(theta):
    ''' rotation matrix given theta
    Inputs:
        theta    - theta (in degrees)
    '''
    theta = np.radians(theta)
    A = [[np.math.cos(theta), -np.math.sin(theta)],
         [np.math.sin(theta), np.math.cos(theta)]]
    return np.array(A)
