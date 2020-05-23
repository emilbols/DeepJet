import tensorflow as tf

# Define custom losses here and add them to the global_loss_list dict (important!)
global_loss_list = {}


def huber_loss(y_true, y_pred, clip_delta=1.0):
    error = y_true - y_pred
    cond  = tf.keras.backend.abs(error) < clip_delta
    squared_loss = 0.5 * tf.keras.backend.square(error)
    linear_loss  = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta)
    return tf.where(cond, squared_loss, linear_loss)

def asym_loss_down(y_true, y_pred, quantile=0.25):
    error = y_true - y_pred
    cond  = error < 0
    low_loss = (1-quantile) * tf.keras.backend.abs(error)
    high_loss  = quantile * tf.keras.backend.abs(error)
    return tf.where(cond, high_loss, low_loss)

def asym_loss_up(y_true, y_pred, quantile=0.75):
    error = y_true - y_pred
    cond  = error < 0
    low_loss = (1-quantile) * tf.keras.backend.abs(error)
    high_loss  = quantile * tf.keras.backend.abs(error)
    return tf.where(cond, high_loss, low_loss)


global_loss_list['huber_loss'] = huber_loss
global_loss_list['asym_loss_down'] = asym_loss_down
global_loss_list['asym_loss_up'] = asym_loss_up
