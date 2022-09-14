from __future__ import print_function

import numpy as np
import camera_calibration as cc
import transformations as tr

def so_project(A):
    """ Project an arbitrary matrix to SO(n) """
    U,S,Vh = np.linalg.svd(A)
    V = Vh.T
    R = U.dot(np.diag([1,1,np.linalg.det(U.dot(V.T))]).dot(V.T))

    return R

def rot3_rand():
    """ Generate a random rotation """
    A = np.random.randn(3,3)
    return so_project(A)

def hat(v):
    v = np.array(v)

    # Check if v is 3-value vector only:
    assert v.size == 3

    V=np.zeros((3,3))
    V[0,1]=-v[2]
    V[1,0]=v[2]
    V[0,2]=v[1]
    V[2,0]=-v[1]
    V[1,2]=-v[0]
    V[2,1]=v[0]

    return V

def exp3(v):
    """ Compute the exponential map of a rotation """
    v = np.array(v)

    # Check if v is 3-value vector only:
    assert v.size == 3

    theta = np.linalg.norm(v)
    
    if abs(theta) < 1e-16:
        return np.eye(3)

    V=np.zeros((3,3))
    V[0,1]=-v[2]
    V[1,0]=v[2]
    V[0,2]=v[1]
    V[2,0]=-v[1]
    V[1,2]=-v[0]
    V[2,1]=v[0]

    #V2 = np.eye(3) * np.cos(theta) + np.sin(theta)/theta * V + (1-np.cos(theta))/np.square(theta) * V.dot(V.T)
    V2 = np.eye(3) + np.sin(theta)/theta * V + (1-np.cos(theta))/np.square(theta) * V.dot(V)
    
    return V2

def log3(R):
    """ Compute log map of rotation """
    R = np.array(R)

    # check if R is 3-by-3 matrix only:
    assert R.shape == (3,3)

    cos = np.max([-1,np.min([1,(np.trace(R)-1)/2])])
    theta = np.arccos(cos)
    
    if theta < 1e-5:
        return np.array([0,0,0])
        
    logR = theta/(2*np.sin(theta)) * (R - R.T)

    return np.array([logR[2,1], logR[0,2], logR[1,0]])

def rot3_randn(sigma):
    """ 
    Sample random rotation from a Gaussian
    distribution around 0 with standard deviation = sigma
    """
    A=exp3(sigma*np.random.randn(3,1))
    return A

def calibration_rotation_measurement(Rx,Rz):
    """ Generate random Ra,Rb such that Ra*Rx=Rz*Rb """
    Rb=rot3_rand()
    Ra=Rz.dot(Rb.dot(Rx.transpose()))
    return Ra,Rb

def calibration_rotation_measurement_randn(Ra,Rb,sigma):
    """ Generate random rotation calibration measurements """
    return rot3_randn(sigma).dot(Ra), rot3_randn(sigma).dot(Rb)

def calibration_translation_measurement(Ra,Rz,tx,tz):
    tb = np.random.rand(3,1)
    ta = Rz.dot(tb) + tz - Ra.dot(tx)
    return ta, tb

def calibration_translation_measurement_randn(ta,tb,sigma):
    return ta + sigma*np.random.randn(3,1), tb + sigma*np.random.randn(3,1)

def calibration_rotation_residual(Ra,Rb,Rx,Rz):
    """ Return the residual of the rotation calibration relation """
    return Ra.dot(Rx)-Rz.dot(Rb)

def rotation_calibration_random_measurement_test(sigma=0.0):
    Rx = rot3_rand()
    Rz = rot3_rand()
    for i_trial in range(0,5):
        Ra,Rb = calibration_rotation_measurement(Rx,Rz)
        Ra,Rb = calibration_rotation_measurement_randn(Ra,Rb,sigma)
        print(calibration_rotation_residual(Ra,Rb,Rx,Rz))


if __name__ == "__main__":
    
    rotation_calibration_random_measurement_test(sigma=1.0e-3)
