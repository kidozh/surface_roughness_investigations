from keras.callbacks import TensorBoard, ModelCheckpoint
from model import build_multi_input_main_residual_network

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# plt.rcParams['font.sans-serif'] = ['Hiragino Sans GB']  # 用来正常显示中文标签
# # plt.rcParams['font.sans-serif'] = ['YaHei Consolas Hybrid']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42

from data import dataSet

data = dataSet()
signal, sf_roughness = data.get_reinforced_data()

print(signal.shape, sf_roughness.shape)

# y = data.gen_y_dat()

# y = data.get_rul_dat()

import random

index = [i for i in range(sf_roughness.shape[0])]
random.shuffle(index)
y = sf_roughness[index]
x = signal[index]

# y = data.get_rul_dat()

# reshape y


for i in [20, 15, 10, 5, 35]:
    DEPTH = i

    log_dir = 'Resnet_down_sample_logs/'

    train_name = 'Resnet_block_%s' % (DEPTH)
    MODEL_CHECK_PT = "%s.kerascheckpts" % (train_name)

    model_name = '%s.kerasmodel' % (train_name)

    predict = True
    model = build_multi_input_main_residual_network(32, 500, 8, 1, loop_depth=DEPTH)
    if not predict:
        tb_cb = TensorBoard(log_dir=log_dir + train_name)
        ckp_cb = ModelCheckpoint(MODEL_CHECK_PT, monitor='val_loss', save_weights_only=True, verbose=1,
                                 save_best_only=True, period=5)

        # model = build_simple_rnn_model(5000,7,3)
        import os.path

        if os.path.exists(MODEL_CHECK_PT):
            model.load_weights(MODEL_CHECK_PT)

        print('Model has been established.')

        model.fit(x, y, batch_size=16, epochs=1000, callbacks=[tb_cb, ckp_cb], validation_split=0.2)

        model.save(model_name)

    else:

        PRED_PATH = 'Y_PRED'

        model.load_weights(MODEL_CHECK_PT)

        # print(model.evaluate(x, y))
        signal, srf = data.get_test_show_data()
        srf_pred = model.predict(signal)
        print(model.metrics_names,model.evaluate(signal,srf))

        fig = plt.figure()

        plt.plot(srf, label="Real Surface Roughness")
        plt.plot(srf_pred, label="Predicted Surface Roughness")
        plt.xlabel("Run")
        plt.ylabel("Surface Roughness ($\mu m$)")
        plt.legend()
        plt.savefig("RESULT.svg")
        fig.show()
        plt.show()

    break
