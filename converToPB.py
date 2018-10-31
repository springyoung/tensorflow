from keras import backend as K
from keras.models import Model, load_model
import numpy as np
import tensorflow as tf
K.set_learning_phase(0)  # all new operations will be in test mode from now on

def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


previous_model = load_model('model_factor_conver.h5', custom_objects={'mean_iou': mean_iou})
previous_model.summary()
# serialize the model and get its weights, for quick re-building
config = previous_model.get_config()
weights = previous_model.get_weights()

# re-build a model where the learning phase is now hard-coded to 0
from keras.models import model_from_config
new_model = Model.from_config(config)
new_model.set_weights(weights)


from tensorflow_serving.session_bundle import exporter

export_path = './' # where to save the exported graph
export_version = 1 # version number (integer)

saver = tf.train.Saver(sharded=True)
model_exporter = exporter.Exporter(saver)
signature = exporter.classification_signature(input_tensor=new_model.input,
                                              scores_tensor=new_model.output)
model_exporter.init(sess.graph.as_graph_def(),
                    default_graph_signature=signature)
model_exporter.export(export_path, tf.constant(export_version), sess)