import tensorflow as tf
from model.model import create_model
from model.input_fn import train_input_fn
from model.loss import composite_loss
from model.utils import Vocabulary
import os,json


tf.compat.v1.disable_eager_execution()


ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
STATIC_CONFIG = dict(json.load(open('config.json','r')))
print(staticmethod)
RUNTIME_CONFIG = {"root_path":ROOT_PATH}
print(RUNTIME_CONFIG)
CONFIG = {**STATIC_CONFIG, **RUNTIME_CONFIG}

vocabulary = Vocabulary()

next_training_batch = train_input_fn(vocabulary, CONFIG)

mel_outputs, residual_mels = create_model(next_training_batch, CONFIG, is_training=True)

loss = composite_loss(mel_outputs, residual_mels, next_training_batch['targets'])

opt = tf.compat.v1.train.AdamOptimizer()

opt_op = opt.minimize(loss)

sess = tf.compat.v1.Session()

merged = tf.compat.v1.summary.merge_all()

train_writer = tf.compat.v1.summary.FileWriter(ROOT_PATH+'/logs' + '/train', sess.graph)

saver = tf.compat.v1.train.Saver()

sess.run(tf.compat.v1.global_variables_initializer())

steps = 0

while True:
    steps += 1
    try:
        ntb, training_loss, summary, _ = sess.run([next_training_batch, loss, merged, opt_op])
        train_writer.add_summary(summary, steps)
        print("-------OUTPUT---------")
        print("Loss {} at batch {}".format(training_loss, steps))
        print("----INPUT BATCH DETAILS------")
        for key, value in ntb.items():
            print('{} - {}'.format(key, value.shape))

        if steps % 10 == 0:
            saver.save(sess, './logs/tacotron-2-explained', global_step=steps)

    except tf.errors.OutOfRangeError:
        saver.save(sess, './logs/tacotron-2-explained-final', global_step=steps)
        print("----TRAINING OVER------")
        break
