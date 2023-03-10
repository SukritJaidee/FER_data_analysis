import tensorflow as tf
import tensorflow_addons as tfa
from keras import backend as K

## section1
def recall_1(y_true, y_pred):
    y_true = K.ones_like(y_true)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (all_positives + K.epsilon())
    return recall

def precision_1(y_true, y_pred):
    y_true = K.ones_like(y_true)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_score_1(y_true, y_pred):
    precision = precision_1(y_true, y_pred)
    recall = recall_1(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# y_true = np.array([[1, 1, 1],
#                             [1, 0, 0],
#                             [1, 1, 0]], np.float32)
# y_pred = np.array([[0.2, 0.6, 0.7],
#                             [0.2, 0.6, 0.6],
#                             [0.6, 0.8, 0.0]], np.float32)
# res = f1_score_1(y_true, y_pred).numpy()
# print(res)

## section2
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (all_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# y_true = np.array([[1, 1, 1],
#                             [1, 0, 0],
#                             [1, 1, 0]], np.float32)
# y_pred = np.array([[0.2, 0.6, 0.7],
#                             [0.2, 0.6, 0.6],
#                             [0.6, 0.8, 0.0]], np.float32)
# res = f1_m(y_true, y_pred)
# print(res.numpy())

## section3
## @title tfa.metrics.F1Score (# Multilabel)

# metric = tfa.metrics.F1Score(num_classes=5, threshold=0.5)
# y_true = np.array([[0, 1, 2, 0, 1]], np.int32)
# y_pred = np.array([[0, 2, 1, 0, 0]], np.float32)
# metric.update_state(y_true, y_pred)
# result = metric.result()
# res1 = result.numpy()
# res2 = res1.mean()
# print(f"tfa res1 {res1} res2 {res2}")

# metric = tfa.metrics.F1Score(num_classes=3, threshold=0.5)
# y_true = np.array([[1, 1, 1],
#                             [1, 0, 0],
#                             [1, 1, 0]], np.int32)
# y_pred = np.array([[0.2, 0.6, 0.7],
#                             [0.2, 0.6, 0.6],
#                             [0.6, 0.8, 0.0]], np.float32)
# metric.update_state(y_true, y_pred)
# result = metric.result()
# res1 = result.numpy()
# res2 = res1.mean()
# print(f"tfa res1 {res1} res2 {res2}")

## section4
#@title f1
def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)
    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())
    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

def f1_loss(y_true, y_pred):
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)
    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())
    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)

# # y_true = np.array([[0, 1, 2, 0, 1]])
# # y_pred = np.array([[0, 2, 1, 0, 0]])
# y_true = np.array([[1, 1, 1], [1, 0, 0], [1, 1, 0]], np.int32)
# y_pred = np.array([[0.2, 0.6, 0.7], [0.2, 0.6, 0.6], [0.6, 0.8, 0.0]], np.float32)
# res1 = f1(y_true, y_pred)
# res2 = res1.numpy()
# print(f"MultilabelF1Score: {res2}")
# res1 = f1_loss(y_true, y_pred)
# print(f"f1 loss: {res1}")




