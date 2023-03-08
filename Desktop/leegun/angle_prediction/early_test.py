import os
import numpy as np
from pyparsing import Char
from Parameters import *
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import LearningRateScheduler
from scipy.ndimage import gaussian_filter1d
import argparse

def make_parser():
    parser = argparse.ArgumentParser("Counting algorithm")
    #gpu_num
    parser.add_argument(
        "-g", "--gpu_num",
        type=str,
        default='0',
        help="Input gpu_number"
    )
    #filter_var
    parser.add_argument(
        "-v", "--filter_variation",
        type=int,
        default=10,
        help="Input filter_variation"
    )
    #learning_rate
    parser.add_argument(
        "-lr", "--learning_rate",
        type = float,
        default=0.97,
        help="Input learning_rate"
    )
    #num_case
    parser.add_argument(
        "-n", "--num_case",
        type = int,
        default=1,
        help="Input num_case"
    )
    #epoch
    parser.add_argument(
        "-e", "--epoch",
        type = int,
        default=100,
        help="Input epoch"
    )
    #dimension
    parser.add_argument(
        "-d", "--dimension",
        type = int,
        default=11,
        help="Input dimension"
    )
    #network_model
    parser.add_argument(
        "-m", "--network_model",
        type = Char,
        default='F',
        help="Input network_model\nFC_network = F\nConv_network = C\nAlex_network = A\nLe_network = L\nVgg-19_network = V\nResnet18_network = R"
    )
    return parser
if __name__ == "__main__" :
    args = make_parser().parse_args()
    gpu_num = args.gpu_num
    filter_var  = args.filter_variation
    learning_rate = args.learning_rate
    num_case =args.num_case
    epoch = args.epoch
    dimension = args.dimension
    model = args.network_model
# import wandb

# wandb.init(project="test", entity="leegun4488")

# wandb.config = {
#   "learning_rate": 0.001,
#   "epochs": 100,
#   "batch_size": 2000
# }

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num

print("dimension = ", dimension)
nBin = 256
print("filter_var = ", filter_var)
print("learning_rate = ", learning_rate)
print("num_case = ", num_case)
print("epoch = ", epoch)

def data_loader(dimension, nBin, filter_var):
    
    anglePerBin = 2*np.pi/nBin
    
    path = [None] * 20
    path[0] = '/lushtoner/abc/Datasets/19-05-30_C10_roomba_PartronCP_DB_Broadspec_01'
    path[1] = '/lushtoner/abc/Datasets/19-05-30_C10_roomba_PartronCP_DB_Broadspec_02'
    path[2] = '/lushtoner/abc/Datasets/19-05-30_C10_roomba_PartronCP_DB_Broadspec_03'
    path[3] = '/lushtoner/abc/Datasets/19-05-30_C10_roomba_PartronCP_DB_Broadspec_04'
    path[4] = '/lushtoner/abc/Datasets/19-05-30_C10_roomba_PartronCP_DB_Broadspec_05'
    path[5] = '/lushtoner/abc/Datasets/19-05-30_C10_roomba_PartronCP_DB_Broadspec_06'
    path[6] = '/lushtoner/abc/Datasets/19-05-30_C10_roomba_PartronCP_DB_Broadspec_07'
    path[7] = '/lushtoner/abc/Datasets/19-05-30_C10_roomba_PartronCP_DB_Broadspec_08'
    path[8] = '/lushtoner/abc/Datasets/19-05-30_C10_roomba_PartronCP_DB_Broadspec_09'
    path[9] = '/lushtoner/abc/Datasets/19-05-30_C10_roomba_PartronCP_DB_Broadspec_10'
    path[10] = '/lushtoner/abc/Datasets/19-06-05_C10_roomba_ModSpline_DB_Broadspec_01'
    path[11] = '/lushtoner/abc/Datasets/19-06-05_C10_roomba_ModSpline_DB_Broadspec_02'
    path[12] = '/lushtoner/abc/Datasets/19-06-05_C10_roomba_ModSpline_DB_Broadspec_03'
    path[13] = '/lushtoner/abc/Datasets/19-06-05_C10_roomba_ModSpline_DB_Broadspec_04'
    path[14] = '/lushtoner/abc/Datasets/19-06-05_C10_roomba_ModSpline_DB_Broadspec_05'
    path[15] = '/lushtoner/abc/Datasets/19-06-05_C10_roomba_ModSpline_DB_Broadspec_06'
    path[16] = '/lushtoner/abc/Datasets/19-06-05_C10_roomba_ModSpline_DB_Broadspec_07'
    path[17] = '/lushtoner/abc/Datasets/19-06-05_C10_roomba_ModSpline_DB_Broadspec_08'
    path[18] = '/lushtoner/abc/Datasets/19-06-05_C10_roomba_ModSpline_DB_Broadspec_09'
    path[19] = '/lushtoner/abc/Datasets/19-06-05_C10_roomba_ModSpline_DB_Broadspec_10'
    
    # train_data #
    
    data = np.empty((0,dimension))
    data_real = np.empty((0,dimension))
    data_imag = np.empty((0,dimension))
    GT = []

    for i in range(8):
        npzfile = np.load(os.path.join(path[i], "processedData.npz"))
        
        # extract CIR
        realCIR = npzfile['realCIR'].astype(dtype=np.float32)
        imagCIR = npzfile['imagCIR'].astype(dtype=np.float32)
        # extract AOA
        AOA = npzfile['yaw_m_mc'].astype(dtype=np.float32)
        # calculate magnitude
        magCIR = np.sqrt(realCIR ** 2 + imagCIR** 2)
        AOA = AOA + np.pi
        np.divide(AOA, anglePerBin, AOA)
        
        data = np.append(data, magCIR[:,4:4+dimension], axis=0)
        data_real = np.append(data_real, realCIR[:,4:4+dimension], axis=0)
        data_imag = np.append(data_imag, imagCIR[:,4:4+dimension], axis=0)
        GT = np.append(GT, AOA, axis=0)

    tan = np.arctan2(data_imag, data_real)
    diff = tan[:,:dimension-1] - tan[:,1:]
    data = np.stack((data, tan), axis=-1)
    
    GT = GT.astype('uint8')
    GT = np.eye(nBin)[GT]
    
    # test_data #
    
    X_test = np.empty((0,dimension))
    y_test = []
    X_test_real = np.empty((0,dimension))
    X_test_imag = np.empty((0,dimension))

    npzfile = np.load(os.path.join(path[8], "processedData.npz"))

    # extract CIR
    realCIR = npzfile['realCIR'].astype(dtype=np.float32)
    imagCIR = npzfile['imagCIR'].astype(dtype=np.float32)
    # extract AOA
    AOA = npzfile['yaw_m_mc'].astype(dtype=np.float32)
    # calculate magnitude
    magCIR = np.sqrt(realCIR ** 2 + imagCIR** 2)
    AOA = AOA + np.pi
    np.divide(AOA, anglePerBin, AOA)
    AOA = AOA.astype(int)
        
    X_test = np.append(X_test, magCIR[:,4:4+dimension], axis=0)
    X_test_real = np.append(X_test_real, realCIR[:,4:4+dimension], axis=0)
    X_test_imag = np.append(X_test_imag, imagCIR[:,4:4+dimension], axis=0)
    y_test = np.append(y_test, AOA, axis=0)

    tan = np.arctan2(X_test_imag, X_test_real)
    diff = tan[:,:dimension-1] - tan[:,1:]
    X_test = np.stack((X_test, tan), axis=-1)

    y_test = y_test.astype('uint8')
    y_test = np.eye(nBin)[y_test]
    
    # X_train, X_val, y_train, y_val = train_test_split(data, GT, test_size=0.2, random_state=1)
    # y_val = np.eye(nBin)[y_val]
    # y_val = gaussian_filter1d(y_val, filter_var, mode='wrap')
    
    if(filter_var):    
        GT = gaussian_filter1d(GT, filter_var, mode='wrap')
        # y_train = gaussian_filter1d(y_train, filter_var, mode='wrap')
    return data, GT, X_test, y_test

data, GT, X_test, y_test = data_loader(dimension, nBin, filter_var)

if model == 'F':
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(dimension,2)),
        tf.keras.layers.Dense(800, activation=tf.nn.relu),
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
elif model == 'C':
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(128,3, activation=tf.nn.relu,input_shape=(dimension,2),padding='same'),
        tf.keras.layers.LayerNormalization(axis=-1),
        tf.keras.layers.Conv1D(256,3, activation=tf.nn.relu,padding='same'),
        tf.keras.layers.LayerNormalization(axis=-1),
        tf.keras.layers.MaxPooling1D(2,padding='same'),
        tf.keras.layers.LayerNormalization(axis=-1),
        tf.keras.layers.Conv1D(512,3, activation=tf.nn.relu,padding='same'),
        tf.keras.layers.LayerNormalization(axis=-1),
        tf.keras.layers.MaxPooling1D(2,padding='same'),
        tf.keras.layers.LayerNormalization(axis=-1),
        tf.keras.layers.Conv1D(1024,3, activation=tf.nn.relu,padding='same'),
        tf.keras.layers.LayerNormalization(axis=-1),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.05),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(nBin, activation='softmax')
    ])
elif model == 'A':
    model = tf.keras.Sequential([
        # layer1
        tf.keras.layers.Conv1D(96,11, activation=tf.nn.relu,input_shape=(dimension,2),padding='same'),
        tf.keras.layers.MaxPooling1D(2,padding='same'),
        # layer2
        tf.keras.layers.Conv1D(256,5, activation=tf.nn.relu,padding='same'),
        tf.keras.layers.MaxPooling1D(2,padding='same'),
        # layer3
        tf.keras.layers.ZeroPadding1D(1),
        tf.keras.layers.Conv1D(384,3, activation=tf.nn.relu,padding='same'),
        # layer4
        tf.keras.layers.ZeroPadding1D(1),
        tf.keras.layers.Conv1D(384,3, activation=tf.nn.relu,padding='same'),
        # layer5
        tf.keras.layers.ZeroPadding1D(1),
        tf.keras.layers.Conv1D(256,3, activation=tf.nn.relu,padding='same'),
        tf.keras.layers.Dropout(0.5),
        # layer6
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        # layer7
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        # layer8
        tf.keras.layers.Dense(nBin, activation='softmax')
    ])
elif model == 'L':
    model = tf.keras.Sequential([
        # layer1
        tf.keras.layers.Conv1D(6,5, activation=tf.nn.relu,input_shape=(dimension,2),padding='same'),
        # layer2
        tf.keras.layers.AveragePooling1D(2,padding='same'),
        # layer3
        tf.keras.layers.Conv1D(16,5, activation=tf.nn.relu,padding='same'),
        # layer4
        tf.keras.layers.AveragePooling1D(2,padding='same'),
        # layer5
        tf.keras.layers.Conv1D(120,5, activation=tf.nn.relu,input_shape=(64,2),padding='same'),
        # layer6
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(840, activation='relu'),
        # layer7
        tf.keras.layers.Dense(nBin, activation='softmax'),
    ])
elif model == 'V':
    model = tf.keras.Sequential([
        # layer1~3
        tf.keras.layers.Conv1D(64,3, activation=tf.nn.relu,input_shape=(dimension,2),padding='same'),
        tf.keras.layers.Conv1D(64,3, activation=tf.nn.relu,padding='same'),
        tf.keras.layers.AveragePooling1D(2,padding='same'),
        # layer4~6
        tf.keras.layers.Conv1D(128,3, activation=tf.nn.relu,padding='same'),
        tf.keras.layers.Conv1D(128,3, activation=tf.nn.relu,padding='same'),
        tf.keras.layers.AveragePooling1D(2,padding='same'),
        # layer7~11
        tf.keras.layers.Conv1D(256,3, activation=tf.nn.relu,padding='same'),
        tf.keras.layers.Conv1D(256,3, activation=tf.nn.relu,padding='same'),
        tf.keras.layers.Conv1D(256,3, activation=tf.nn.relu,padding='same'),
        tf.keras.layers.Conv1D(256,3, activation=tf.nn.relu,padding='same'),
        tf.keras.layers.AveragePooling1D(2,padding='same'),
        # layer12~16
        tf.keras.layers.Conv1D(512,3, activation=tf.nn.relu,padding='same'),
        tf.keras.layers.Conv1D(512,3, activation=tf.nn.relu,padding='same'),
        tf.keras.layers.Conv1D(512,3, activation=tf.nn.relu,padding='same'),
        tf.keras.layers.Conv1D(512,3, activation=tf.nn.relu,padding='same'),
        tf.keras.layers.AveragePooling1D(2,padding='same'),
        # layer17~21
        tf.keras.layers.Conv1D(512,3, activation=tf.nn.relu,padding='same'),
        tf.keras.layers.Conv1D(512,3, activation=tf.nn.relu,padding='same'),
        tf.keras.layers.Conv1D(512,3, activation=tf.nn.relu,padding='same'),
        tf.keras.layers.Conv1D(512,3, activation=tf.nn.relu,padding='same'),
        tf.keras.layers.AveragePooling1D(2,padding='same'),
        # layer22~24
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation=tf.nn.relu),
        tf.keras.layers.Dense(4096, activation=tf.nn.relu),
        tf.keras.layers.Dense(nBin, activation='softmax'),
    ])
elif model == 'R':
    class ResNet18(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.convA1 = tf.keras.layers.Conv1D(64,7,2,activation=tf.nn.relu,input_shape=(dimension-1,2),padding='same')
            self.normA1 = tf.keras.layers.BatchNormalization()
            self.poolA1 = tf.keras.layers.MaxPooling1D(2,padding='same')

            self.convB1 = tf.keras.layers.Conv1D(64,3,1,padding='same')
            self.convB2 = tf.keras.layers.Conv1D(64,3,1,padding='same')
            self.convB3 = tf.keras.layers.Conv1D(64,3,1,padding='same')
            self.convB4 = tf.keras.layers.Conv1D(64,3,1,padding='same')
            self.normB1 = tf.keras.layers.BatchNormalization()
            self.normB2 = tf.keras.layers.BatchNormalization()
            self.poolB1 = tf.keras.layers.MaxPooling1D(2,padding='same')

            self.convC1 = tf.keras.layers.Conv1D(128,3,1,padding='same')
            self.convC2 = tf.keras.layers.Conv1D(128,3,1,padding='same')
            self.convC3 = tf.keras.layers.Conv1D(128,3,1,padding='same')
            self.convC4 = tf.keras.layers.Conv1D(128,3,1,padding='same')
            self.normC1 = tf.keras.layers.BatchNormalization()
            self.normC2 = tf.keras.layers.BatchNormalization()
            self.poolC1 = tf.keras.layers.MaxPooling1D(2,padding='same')
            self.shortcutC = tf.keras.layers.Conv1D(128,1,1,padding='same')

            self.convD1 = tf.keras.layers.Conv1D(256,3,1,padding='same')
            self.convD2 = tf.keras.layers.Conv1D(256,3,1,padding='same')
            self.convD3 = tf.keras.layers.Conv1D(256,3,1,padding='same')
            self.convD4 = tf.keras.layers.Conv1D(256,3,1,padding='same')
            self.normD1 = tf.keras.layers.BatchNormalization()
            self.normD2 = tf.keras.layers.BatchNormalization()
            self.poolD1 = tf.keras.layers.MaxPooling1D(2,padding='same')
            self.shortcutD = tf.keras.layers.Conv1D(256,1,1,padding='same')

            # self.convE1 = tf.keras.layers.Conv1D(512,3,1,padding='same')
            # self.convE2 = tf.keras.layers.Conv1D(512,3,1,padding='same')
            # self.convE3 = tf.keras.layers.Conv1D(512,3,1,padding='same')
            # self.convE4 = tf.keras.layers.Conv1D(512,3,1,padding='same')
            # self.normE1 = tf.keras.layers.BatchNormalization()
            # self.normE2 = tf.keras.layers.BatchNormalization()
            # self.poolE1 = tf.keras.layers.MaxPooling1D(2,padding='same')
            # self.shortcutE = tf.keras.layers.Conv1D(512,1,1,padding='same')

            self.normF = tf.keras.layers.AveragePooling1D(2,padding='same')
            self.flatten = tf.keras.layers.Flatten()
            self.denseF1 = tf.keras.layers.Dense(1000,activation=tf.nn.relu)
            self.denseF2 = tf.keras.layers.Dense(nBin,activation=tf.nn.softmax)
        def call(self, inputs):
            # blockA
            x = self.convA1(inputs)
            x = self.normA1(x)
            x = self.poolA1(x)
            # blockB
            con_b = self.convB1(x)
            con_b = self.normB1(con_b)
            con_b = tf.keras.activations.relu(con_b)
            con_b = self.convB2(con_b)
            x += con_b
            x = tf.keras.activations.relu(x)
            con_b = self.convB3(x)
            con_b = self.normB2(con_b)
            con_b = tf.keras.activations.relu(con_b)
            con_b = self.convB4(con_b)
            x += con_b
            x = tf.keras.activations.relu(x)
            x = self.poolB1(x)
            # blockC
            con_c = self.convC1(x)
            con_c = self.normC1(con_c)
            con_c = tf.keras.activations.relu(con_c)
            con_c = self.convC2(con_c)
            x = self.shortcutC(x)
            x += con_c
            x = tf.keras.activations.relu(x)
            con_c = self.convC3(x)
            con_c = self.normC2(con_c)
            con_c = tf.keras.activations.relu(con_c)
            con_c = self.convC4(con_c)
            x += con_c
            x = tf.keras.activations.relu(x)
            x = self.poolC1(x)
            # blockD
            con_d = self.convD1(x)
            con_d = self.normD1(con_d)
            con_d = tf.keras.activations.relu(con_d)
            con_d = self.convD2(con_d)
            x = self.shortcutD(x)
            x += con_d
            x = tf.keras.activations.relu(x)
            con_d = self.convD3(x)
            con_d = self.normD2(con_d)
            con_d = tf.keras.activations.relu(con_d)
            con_d = self.convD4(con_d)
            x += con_d
            x = tf.keras.activations.relu(x)
            x = self.poolD1(x)
            # # blockE
            # con_e = self.convE1(x)
            # con_e = self.normE1(con_e)
            # con_e = tf.keras.activations.relu(con_e)
            # con_e = self.convE2(con_e)
            # x = self.shortcutE(x)
            # x += con_e
            # x = tf.keras.activations.relu(x)
            # con_e = self.convE3(x)
            # con_e = self.normE2(con_e)
            # con_e = tf.keras.activations.relu(con_e)
            # con_e = self.convE4(con_e)
            # x += con_e
            # x = tf.keras.activations.relu(x)
            # blockF
            x = self.normF(x)
            x = self.flatten(x)
            x = self.denseF1(x)
            x = self.denseF2(x)
            return x
    model = ResNet18()
    model.build(input_shape = (None,dimension,2))

print(model.summary())

def step_decay(epoch):
    start = 0.01
    drop = learning_rate
    epochs_drop = 5.0
    lr = start * (drop ** np.floor((epoch)/epochs_drop))
    return lr

lr_scheduler = LearningRateScheduler(step_decay, verbose=1)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.001),
              loss="categorical_crossentropy",
              metrics=['accuracy'])
history = model.fit(data, GT, batch_size = 2000, epochs=epoch, callbacks=[lr_scheduler])

# , validation_data = (X_val, y_val) #

results = model.evaluate(X_test, y_test, batch_size = 2000)

y_test = np.argmax(y_test, axis = 1)

if(num_case==1):
    pred_sample = model.predict(X_test)
    pred_index = np.argmax(pred_sample, axis = 1)

    pred_index = pred_index + 15
    y_test = y_test + 15

    TF = (y_test >= pred_index-15) & (y_test <= pred_index+15)
    num_true = np.where(TF == True)[0].shape[0]
    num_all = y_test.shape[0]
    print("accuracy :",num_true / num_all * 100 , "%")
else:
    test_data = {}
    for i in range(nBin):
        test_data[i] = np.empty((0,dimension,2))

    for k in range(y_test.shape[0]):
        temp = X_test[k,:].reshape(1,dimension,2)
        test_data[y_test[k]] = np.vstack((temp, test_data[y_test[k]]))

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
        for j in range(int(test_data[0].shape[0]/num_case)):
            sum_data = np.zeros(nBin)
            sum_data = np.sum(pred_sample[j*num_case:(j+1)*num_case, :], axis = 0)
            if (downswitch + upswitch) == 0:
                if down_boundary < np.argmax(sum_data) and np.argmax(sum_data) < up_boundary:
                    cnt = cnt + 1
            elif downswitch == 1:
                if down_boundary < np.argmax(sum_data) or np.argmax(sum_data) < up_boundary:
                    cnt = cnt + 1
            elif upswitch == 1:
                if down_boundary < np.argmax(sum_data) or np.argmax(sum_data) < up_boundary:
                    cnt = cnt + 1
        print(i, 'accuracy : ', cnt/int(test_data[0].shape[0]/num_case)*100)
        all_accuracy = all_accuracy + cnt/int(test_data[0].shape[0]/num_case)*100
    print('all accuracy', all_accuracy/nBin)

# with tf.Session() as sess:
#   # ...
#   wandb.tensorflow.log(tf.summary.merge_all())