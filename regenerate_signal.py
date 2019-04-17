from dcgan import DCGAN,normalize_sample_to_signal
from model import build_multi_input_main_residual_network
from data import dataSet
import numpy as np
import matplotlib.pyplot as plt


# get condition labels
data = dataSet()
condition_labels = data.get_condition_number_data()
# load the model
dcgan = DCGAN()
# not specify the epoch
EPOCH = None
dcgan.load_model(epoch=EPOCH)
print(condition_labels.shape)


noise = np.random.normal(0, 1, (condition_labels.shape[0], dcgan.latent_dim))
gen_imgs = dcgan.generator.predict([noise, condition_labels])
gen_imgs = normalize_sample_to_signal(gen_imgs)
model = build_multi_input_main_residual_network(32, 500, 8, 1, loop_depth=20)
train_name = 'Resnet_block_REDUCE_AE_%s' % (20)
MODEL_CHECK_PT = "%s.kerascheckpts" % (train_name)
model.load_weights(MODEL_CHECK_PT)

# print(model.evaluate(x, y))
signal, srf = data.get_test_show_data()
srf_pred = model.predict(gen_imgs)
print(model.metrics_names,model.evaluate(signal,srf))

fig = plt.figure()

plt.plot(srf, label="Real Surface Roughness")
plt.plot(srf_pred, label="Predicted Surface Roughness")
plt.xlabel("Run")
plt.ylabel("Surface Roughness ($\mu m$)")
plt.legend()
plt.savefig("SIMULATE_RESULT.svg")
fig.show()
plt.show()

