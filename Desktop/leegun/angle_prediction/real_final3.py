import os
import numpy as np
# import matplotlib.pylab as plt
import glob
from Parameters import *
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.ndimage import gaussian_filter1d
from tensorflow.keras.callbacks import LearningRateScheduler

numMeas = 20
dimention = 11
nBin = 256
anglePerBin = 2*np.pi/nBin
AOAThreshold = anglePerBin
filter_var = 5

pythonFilePath = os.getcwd()
path = glob.glob(os.path.join(pythonFilePath, 'Downloads/abc','Datasets', '*'))
path[0] = '/lushtoner/Downloads/abc/Datasets/19-05-30_C10_roomba_PartronCP_DB_Broadspec_01'
path[1] = '/lushtoner/Downloads/abc/Datasets/19-05-30_C10_roomba_PartronCP_DB_Broadspec_02'
path[2] = '/lushtoner/Downloads/abc/Datasets/19-05-30_C10_roomba_PartronCP_DB_Broadspec_03'
path[3] = '/lushtoner/Downloads/abc/Datasets/19-05-30_C10_roomba_PartronCP_DB_Broadspec_04'
path[4] = '/lushtoner/Downloads/abc/Datasets/19-05-30_C10_roomba_PartronCP_DB_Broadspec_05'
path[5] = '/lushtoner/Downloads/abc/Datasets/19-05-30_C10_roomba_PartronCP_DB_Broadspec_06'
path[6] = '/lushtoner/Downloads/abc/Datasets/19-05-30_C10_roomba_PartronCP_DB_Broadspec_07'
path[7] = '/lushtoner/Downloads/abc/Datasets/19-05-30_C10_roomba_PartronCP_DB_Broadspec_08'
path[8] = '/lushtoner/Downloads/abc/Datasets/19-05-30_C10_roomba_PartronCP_DB_Broadspec_09'
path[9] = '/lushtoner/Downloads/abc/Datasets/19-05-30_C10_roomba_PartronCP_DB_Broadspec_10'
path[10] = '/lushtoner/Downloads/abc/Datasets/19-06-05_C10_roomba_ModSpline_DB_Broadspec_01'
path[11] = '/lushtoner/Downloads/abc/Datasets/19-06-05_C10_roomba_ModSpline_DB_Broadspec_02'
path[12] = '/lushtoner/Downloads/abc/Datasets/19-06-05_C10_roomba_ModSpline_DB_Broadspec_03'
path[13] = '/lushtoner/Downloads/abc/Datasets/19-06-05_C10_roomba_ModSpline_DB_Broadspec_04'
path[14] = '/lushtoner/Downloads/abc/Datasets/19-06-05_C10_roomba_ModSpline_DB_Broadspec_05'
path[15] = '/lushtoner/Downloads/abc/Datasets/19-06-05_C10_roomba_ModSpline_DB_Broadspec_06'
path[16] = '/lushtoner/Downloads/abc/Datasets/19-06-05_C10_roomba_ModSpline_DB_Broadspec_07'
path[17] = '/lushtoner/Downloads/abc/Datasets/19-06-05_C10_roomba_ModSpline_DB_Broadspec_08'
path[18] = '/lushtoner/Downloads/abc/Datasets/19-06-05_C10_roomba_ModSpline_DB_Broadspec_09'
path[19] = '/lushtoner/Downloads/abc/Datasets/19-06-05_C10_roomba_ModSpline_DB_Broadspec_10'

data = np.empty((0,dimention))
data_real = np.empty((0,dimention))
data_imag = np.empty((0,dimention))
GT = []

for i in range(9):
    npzfile = np.load(os.path.join(path[i], "processedData.npz"))
    
    # extract CIR
    realCIR = npzfile['realCIR'].astype(dtype=np.float32)
    imagCIR = npzfile['imagCIR'].astype(dtype=np.float32)
    offsetCIR = npzfile['offsetCIR'].astype(dtype=np.float32)
    # extract AOA
    AOA = npzfile['yaw_m_mc'].astype(dtype=np.float32)
    # calculate magnitude
    magCIR = np.sqrt(realCIR ** 2 + imagCIR** 2)
    AOA = AOA + np.pi
    for j in range(nBin):
#         std = (j*1.40625-180)*np.pi/180
        std = j*anglePerBin
        angleDiff = AOA-std
        angleDiff = np.where(angleDiff < 0, 2*np.pi, angleDiff)
        AOAThresholdMask = angleDiff<AOAThreshold
        selectionIndices = np.nonzero(AOAThresholdMask)[0][:]
        
        value = magCIR[selectionIndices,4:4+dimention]
        idx = np.ones(value.shape[0])*j
        real = realCIR[selectionIndices,4:4+dimention]
        imag = imagCIR[selectionIndices,4:4+dimention]
        
        data = np.append(data, value, axis = 0)
        GT = np.append(GT, idx)
        data_real = np.append(data_real, real, axis = 0)
        data_imag = np.append(data_imag, imag, axis = 0)

tan = np.arctan2(data_imag, data_real)
diff = tan[:,:10] - tan[:,1:]
data = np.concatenate((data[:,:10], diff), axis=1)

GT = GT.astype('uint8')

# X_train, X_test, y_train, y_test = train_test_split(data, GT, test_size=0.3, shuffle=True, stratify=GT)
X_train, X_val, y_train, y_val = train_test_split(data, GT, test_size=0.2, random_state=1)

y_val = np.eye(nBin)[y_val]
y_train = np.eye(nBin)[y_train]

y_val = gaussian_filter1d(y_val, filter_var, mode='wrap')
y_train = gaussian_filter1d(y_train, filter_var, mode='wrap')

model = tf.keras.Sequential([
    tf.keras.layers.Dense(800, activation=tf.nn.relu, input_shape=(20,)),
    tf.keras.layers.BatchNormalization(axis=-1),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(400, activation=tf.nn.relu),
    tf.keras.layers.BatchNormalization(axis=-1),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(400, activation=tf.nn.relu),
    tf.keras.layers.BatchNormalization(axis=-1),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(nBin, activation='softmax')
])

print(model.summary())

# from tensorflow.keras import backend as K

# def cos_loss(y_true, y_pred):
#     error = y_true - y_pred
#     seta = error * 2 * np.pi / 256
#     loss = tf.math.cos(seta)
#     return loss
# 'SparseCategoricalCrossentropy' , tf.keras.losses.cosine_similarity

def step_decay(epoch):
    start = 0.01
    drop = 0.97
    epochs_drop = 1.0
    lr = start * (drop ** np.floor((epoch)/epochs_drop))
    return lr

lr_scheduler = LearningRateScheduler(step_decay, verbose=1) 

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.01),
              loss="categorical_crossentropy",
              metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size = 2000, epochs=100, validation_data = (X_val, y_val), callbacks=[lr_scheduler])

X_test = np.empty((0,dimention))
y_test = []
X_test_real = np.empty((0,dimention))
X_test_imag = np.empty((0,dimention))

npzfile = np.load(os.path.join(path[9], "processedData.npz"))

# extract CIR
realCIR = npzfile['realCIR'].astype(dtype=np.float32)
imagCIR = npzfile['imagCIR'].astype(dtype=np.float32)
offsetCIR = npzfile['offsetCIR'].astype(dtype=np.float32)
# extract AOA
AOA = npzfile['yaw_m_mc'].astype(dtype=np.float32)
# calculate magnitude
magCIR = np.sqrt(realCIR ** 2 + imagCIR** 2)
AOA = AOA + np.pi

for j in range(nBin):
    std = j*anglePerBin
    angleDiff = AOA-std
    angleDiff = np.where(angleDiff < 0, 2*np.pi, angleDiff)
    AOAThresholdMask = angleDiff<AOAThreshold
    selectionIndices = np.nonzero(AOAThresholdMask)[0][:]
        
    value = magCIR[selectionIndices,4:4+dimention]
    idx = np.ones(value.shape[0])*j
    real = realCIR[selectionIndices,4:4+dimention]
    imag = imagCIR[selectionIndices,4:4+dimention]
        
    X_test = np.append(X_test, value, axis = 0)
    y_test = np.append(y_test, idx)
    X_test_real = np.append(X_test_real, real, axis = 0)
    X_test_imag = np.append(X_test_imag, imag, axis = 0)

tan = np.arctan2(X_test_imag, X_test_real)
diff = tan[:,:10] - tan[:,1:]
X_test = np.concatenate((X_test[:,:10], diff), axis=1)

y_test = y_test.astype('uint8')
y_test = np.eye(nBin)[y_test]
y_test = gaussian_filter1d(y_test, filter_var, mode='wrap')

results = model.evaluate(X_test, y_test, batch_size = 2000)

test_data = {}
for i in range(nBin):
    test_data[i] = np.empty((0,20))

for k in range(y_test.shape[0]):
    temp = X_test[k,:].reshape(1, 20)
    test_data[np.argmax(y_test[k])] = np.vstack((temp, test_data[np.argmax(y_test[k])]))

numcase = 10
all_accuracy = 0

for i in range(nBin):
    pred_sample = model.predict(test_data[i])
    cnt = 0
    downswitch = 0
    upswitch = 0
    down_boundary = i - 10
    up_boundary = i + 10
    if down_boundary < 0:
        down_boundary = down_boundary + nBin
        downswitch = 1
    if up_boundary >= nBin:
        up_boundary = up_boundary - nBin
        upswitch = 1
    for j in range(int(test_data[0].shape[0]/numcase)):
        sum_data = np.zeros(nBin)
        sum_data = np.sum(pred_sample[j*numcase:(j+1)*numcase, :], axis = 0)
        probabilty = np.zeros(nBin)
        for k in range(nBin):
            if k-1 < 0:
                probabilty[k] = np.sum(sum_data[k-10+nBin:]) + np.sum(sum_data[:k+10])
            elif k+1 > nBin:
                probabilty[k] = np.sum(sum_data[k-10:]) + np.sum(sum_data[:k+10-nBin])
            else:
                probabilty[k] = np.sum(sum_data[k-10:k+10])
        if (downswitch + upswitch) == 0:
            if down_boundary < np.argmax(probabilty) and np.argmax(probabilty) < up_boundary:
                cnt = cnt + 1
        elif downswitch == 1:
            if down_boundary < np.argmax(probabilty) or np.argmax(probabilty) < up_boundary:
                cnt = cnt + 1
        elif upswitch == 1:
            if down_boundary < np.argmax(probabilty) or np.argmax(probabilty) < up_boundary:
                cnt = cnt + 1
#         plt.bar(x, sum_data)
#         plt.xticks(x, index)
#         plt.title(i)
#         plt.show()
    print(i, 'accuracy : ', cnt/int(test_data[0].shape[0]/numcase)*100)
    all_accuracy = all_accuracy + cnt/int(test_data[0].shape[0]/numcase)*100
print('all accuracy', all_accuracy/nBin)