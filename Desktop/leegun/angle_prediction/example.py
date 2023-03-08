import sys, os
import numpy as np
import matplotlib.pylab as plt

# get directory of this file
pythonFileName = sys.argv[0]
pythonFilePath = os.path.dirname(pythonFileName)

# get parameters
sys.path.append(pythonFilePath)
from Parameters import *

# load npzfile
npzFilePath = os.path.join(pythonFilePath,"Datasets/19-05-30_C10_roomba_PartronCP_DB_Broadspec_01")
npzfile = np.load(os.path.join(npzFilePath, "processedData.npz"))

# extract CIR
realCIR = npzfile['realCIR'].astype(dtype=np.float32)
imagCIR = npzfile['imagCIR'].astype(dtype=np.float32)
offsetCIR = npzfile['offsetCIR'].astype(dtype=np.float32)


# extract AOA
AOA = npzfile['yaw_m_mc'].astype(dtype=np.float32)


# calculate magnitude
magCIR = np.sqrt(realCIR ** 2 + imagCIR** 2)

# select indices of first 20 CIR measurements received with a similar angle
numMeas = 20
AOAThreshold = 3/180*np.pi
AOAThresholdMask = np.abs(AOA-AOA[0])<AOAThreshold
selectionIndices = np.nonzero(AOAThresholdMask)[0][0:numMeas]

# plot magnitude of these CIR
Ts = 1/(2*FCHIPPING_Hz)

# time axis for CIR measurements in nanoseconds
timeCIR = np.arange(-SAMPLESBEFOREFPINDEX, -SAMPLESBEFOREFPINDEX+ACCMEMLENGTH*Ts*1e9, Ts*1e9, dtype=np.float)
timeCIRSelection = np.expand_dims(timeCIR, axis=0) - np.expand_dims(offsetCIR[selectionIndices], axis=-1)

# plot the magnitude of these measurements
plt.figure()
plt.plot(np.transpose(timeCIRSelection), np.transpose(magCIR[selectionIndices,:]), '.')
plt.xlabel('time (ns)')
plt.ylabel('magnitude (arbitrary unit)')
plt.title('First {:d} CIR measurements received with an AOA of {:.2f} deg'.format(numMeas, AOA[0]*180/np.pi))
plt.show()