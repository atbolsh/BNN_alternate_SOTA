import tensorflow_datasets as tfds
import larq as lq
import tensorflow as tf
import tensorflow.keras as keras
import larq_zoo as lqz
from urllib.request import urlopen
from PIL import Image
import numpy as np
import pandas as pd

import time

from fullRenorm import *
from fullestRenorm import *
from remadeQuicknet import *
from trueControl import *

NUM_EPOCHS = 45# - 35
NUM_TRIALS = 1

# Training script.
# For reasons related to my inexperience with tensorflow, the top-k score is not the correct value; however, the top-1 score is correct.
# Set up to have 2 different learning rates (one for fp, one for binary layers), cosine decay of lr, and several dictionaries that make it
# easy to train several networks with an identically-coded training loop (see the bottome portion).


# imagenette_builder = tfds.builder("imagenette/full-size")
# imagenette_info = imagenette_builder.info
# imagenette_builder.download_and_prepare()
# datasets = imagenette_builder.as_dataset(as_supervised=True)

true_start = time.time()

imagenet_builder = tfds.builder("imagenet2012")
imagenet_ingo = imagenet_builder.info
imagenet_builder.download_and_prepare() 
datasets = imagenet_builder.as_dataset(as_supervised=True)
 
def preprocess(image, label):
    img = lqz.preprocess_input(tf.image.resize(image, (224, 224)))
    return img, label

# Double-check the outputs of the networks one more time, just to make sure it's the right format / size.

train, test = datasets['train'], datasets['validation']

batch_size = 256

train_batch = train.map(
    preprocess).shuffle(32).batch(batch_size) 

validation_batch = test.map(
    preprocess).shuffle(32).batch(batch_size)


class CustomModel(keras.Model):
    def compile(self, opt_fc, opt_bin, loss, metrics):
        self.opt_fc = opt_fc
        self.opt_bin = opt_bin
        super(CustomModel, self).compile(loss=loss, metrics=metrics)
    
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Assign to the two groups, correctly
        bin_vars = []
        fc_vars = []
        bin_grads = []
        fc_grads = []
        for i in range(len(trainable_vars)):
            if 'quantizer' in trainable_vars[i].__dict__.keys():
                bin_vars.append(trainable_vars[i])
                bin_grads.append(gradients[i])
            else:
                fc_vars.append(trainable_vars[i])
                fc_grads.append(gradients[i])
        # Update weights
        self.opt_fc.apply_gradients(zip(fc_grads, fc_vars))
        self.opt_bin.apply_gradients(zip(bin_grads, bin_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


def customFactor(epoch, max_steps = 100 - 5):
    if epoch <= 5:
        return epoch / 5
    else:
        return 0.5*(1 + np.cos(np.pi*min(epoch - 5, max_steps) / max_steps))
    
class CustomCallback(keras.callbacks.Callback):
    def __init__(self, initial_fp_rate = 0.1, initial_bin_rate = 0.01, max_steps = 100):
        super(CustomCallback, self).__init__()
        self.initial_fp_rate = initial_fp_rate
        self.initial_bin_rate = initial_bin_rate
        self.max_steps = max_steps
        
    def on_epoch_begin(self, epoch, logs = None):
        factor = customFactor(epoch+1, self.max_steps - 4) 
        print(factor)
        self.model.opt_fc.learning_rate = self.initial_fp_rate * factor
        self.model.opt_bin.learning_rate = self.initial_bin_rate * factor

#keys = ['control', 'modified', 'fullest', 'remade']
# keys = ['fullest']
keys = ['trueControl']

classNames = { 'control': lqz.sota.QuickNetSmall, # Control
               'modified': ModifiedQuickNetSmall, # Batchnorms in full-precision layers, but not in residuals.
               'fullest': FullestQuickNetSmall,   # No batchnorms, same number of trainable parameters as control.
               'remade': RemadeQuickNetSmall, # No batchnorms, same number of parameters as control, slightly more trainable parameters.
               'trueControl': TrueControlQuickNetSmall} # Same as control, but with maxpool, not maxblurpool. Should be a fairer comparison.

lrpairs = { 'control': (0.1, 0.01),\
            'modified': (0.1, 0.01),\
            'fullest': (0.005, 0.0005),\
            'remade': (0.01, 0.01),\
            'trueControl': (0.1, 0.01)}

wpath = { 'control': "./weights/ImageNet-fullCosine-60epochAttempt-control-epoch{epoch:04d}-trial{trial:02d}.ckpt",\
          'modified': "./weights/ImageNet-modified-epoch{epoch:04d}-trial{trial:02d}.ckpt",\
          'fullest': "./weights/ImageNet-ultimate_LRs-fullCosine-restart-fullest-epoch{epoch:04d}-trial{trial:02d}.ckpt",\
          'remade': "./weights/ImageNet-remade-epoch{epoch:04d}-trial{trial:02d}.ckpt",
          'trueControl': "./weights/ImageNet-fullCosine-45epoch-trueControl-epoch{epoch:04d}-trial{trial:02d}.ckpt"}

lcpath = { 'control': './tabular/ImageNet-fullCosine-60epochAttempt-control-results-trial{trial:02d}.csv',\
           'modified': './tabular/ImageNet-modified-results-trial{trial:02d}.csv',\
           'fullest': './tabular/ImageNet-fullCosine-ultimate_LRs-restart-fullest-results-trial{trial:02d}.csv',\
           'remade': './tabular/ImageNet-remade-results-trial{trial:02d}.csv', 
           'trueControl': "./tabular/ImageNet-fullCosine-45epoch-trueControl-trial{trial:02d}.ckpt"}


# Partial means only after epoch 8
warmup_end = time.time()

print("time to warmup:  " + str(warmup_end - true_start) + "s")

for trial in range(NUM_TRIALS):
    for key in keys:
        model_base = classNames[key](weights=None, num_classes=1000)
        model = CustomModel(model_base.inputs, model_base.outputs)
        # New line after the crash
#        model.load_weights("./weights/ImageNet-ultimate_LRs-fullCosine--fullest-epoch0035-trial00.ckpt")
        # End modification
        fp_rate, bin_rate = lrpairs[key]
        model.compile(opt_fc = tf.keras.optimizers.SGD(learning_rate = fp_rate, momentum=0.9),
                      opt_bin = tf.keras.optimizers.Adam(learning_rate = bin_rate),
                      loss = 'sparse_categorical_crossentropy',
                      metrics=['accuracy', 'top_k_categorical_accuracy']) # Add the top-k before running on ImageNet.
        cc = CustomCallback(max_steps = NUM_EPOCHS, initial_fp_rate = fp_rate, initial_bin_rate = bin_rate) 
        partial_format = wpath[key].split('trial')[0] + ("trial{trial:02d}.ckpt".format(trial=trial))
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=partial_format,
                                                         save_weights_only=True,
                                                         verbose=1)
        history = model.fit(train_batch, validation_data=validation_batch, shuffle=True, epochs=NUM_EPOCHS, callbacks=[cc, cp_callback])
        model_base.save_weights(wpath[key].format(epoch=NUM_EPOCHS,trial=trial))
        results = pd.DataFrame()
        results['train_accuracy'] = history.history['accuracy']
        results['train_top_5'] = history.history['top_k_categorical_accuracy']
        results['val_accuracy'] = history.history['val_accuracy']
        results['val_top_5'] = history.history['val_top_k_categorical_accuracy']
        results.to_csv(lcpath[key].format(trial=trial))

training_over = time.time()
print("training time:   " + str(training_over - warmup_end) + "S")

