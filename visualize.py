"""
#Vikas Bhardwaj

"""
import random

import tensorflow as tf
import dataUtils

TENSORBOARD_LOGDIR = "logdir"

# Creates a batch of random glimpses
def createRandomGlimpses(batch_size):
    glimpses = []
    for i in range(batch_size):
        x_center = random.uniform(-1.0, 1.0)
        y_center = random.uniform(-1.0, 1.0)
        x_width = random.uniform(0.2, 0.4)
        y_width = random.uniform(0.2, 0.4)

        x1 = max(-1.0, min(1.0, x_center - (x_width / 2.0)))
        y1 = max(-1.0, min(1.0, y_center - (y_width / 2.0)))
        x2 = max(-1.0, min(1.0, x_center + (x_width / 2.0)))
        y2 = max(-1.0, min(1.0, y_center + (y_width / 2.0)))
        glimpses.append([y1, x1, y2, x2])

    batch_number = [0 for x in range(batch_size)]
    return glimpses, batch_number

# Clear the old log files
dataUtils.deleteDirectory(TENSORBOARD_LOGDIR)


### Build tensorflow blueprint ###
# Tensorflow placeholder
feature_viz = tf.Variable(tf.random_normal([1, 200, 200, 3], stddev=0.35),
                      name="feature_viz")

# View sample inputs in tensorboard
tf.summary.image("feature_viz", feature_viz)

# Glimpses are x,y coordinates from -1.0 to 1.0
glipmse_offset_placeholder = tf.placeholder(tf.float32, shape=[None, 4])
glipmse_batch_number_placeholder = tf.placeholder(tf.int32, shape=[None])

input_glimpses = tf.image.crop_and_resize(feature_viz, glipmse_offset_placeholder,
                                          glipmse_batch_number_placeholder,
                                          crop_size=[32, 32])

# skip normalizer for visualization
# Normalize image
# Subtract off the mean and divide by the variance of the pixels.
# normalized_image = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), input_glimpses)

'''
# TODO change model to saved model
with tf.variable_scope("Conv_Layer_1"):
    conv_layer_1 = tf.layers.conv2d(input_glimpses,
                                    filters=25,
                                    kernel_size=3,
                                    strides=(1,1),
                                    padding='same',
                                    activation=tf.nn.relu)

    conv_layer_1_with_bn = tf.layers.batch_normalization(conv_layer_1, training=False)


with tf.variable_scope("Conv_Layer_2"):
    conv_layer_2 = tf.layers.conv2d(conv_layer_1_with_bn,
                                    filters=100,
                                    kernel_size=3,
                                    strides=(1,1),
                                    padding='same',
                                    activation=tf.nn.relu)

    conv_layer_2_with_bn = tf.layers.batch_normalization(conv_layer_2, training=False)

    pool_layer_1 = tf.layers.max_pooling2d(conv_layer_2_with_bn,
                                           pool_size=2,
                                           strides=2)

with tf.variable_scope("Conv_Layer_3"):
    conv_layer_3 = tf.layers.conv2d(pool_layer_1,
                                    filters=100,
                                    kernel_size=3,
                                    strides=(1,1),
                                    padding='same',
                                    activation=tf.nn.relu)

    conv_layer_3_with_bn = tf.layers.batch_normalization(conv_layer_3, training=False)

with tf.variable_scope("Conv_Layer_4"):
    conv_layer_4 = tf.layers.conv2d(conv_layer_3_with_bn,
                                    filters=100,
                                    kernel_size=3,
                                    strides=(1, 1),
                                    padding='same',
                                    activation=tf.nn.relu)

    conv_layer_4_with_bn = tf.layers.batch_normalization(conv_layer_4, training=False)

    pool_layer_2 = tf.layers.max_pooling2d(conv_layer_4_with_bn,
                                           pool_size=2,
                                           strides=2)

with tf.variable_scope("Conv_Layer_5"):
    conv_layer_5 = tf.layers.conv2d(pool_layer_2,
                                    filters=100,
                                    kernel_size=3,
                                    strides=(1, 1),
                                    padding='same',
                                    activation=tf.nn.relu)

    conv_layer_5_with_bn = tf.layers.batch_normalization(conv_layer_5, training=False)

    pool_layer_3 = tf.layers.max_pooling2d(conv_layer_5_with_bn,
                                           pool_size=2,
                                           strides=2)


# convert 3d image to 1d tensor (don't change batch dimension)
flat_tensor = tf.contrib.layers.flatten(pool_layer_3)

## Neural network hidden layers
hidden_layer_1 = tf.layers.dense(flat_tensor, 50, activation=tf.nn.relu)
hidden_layer_1_with_bn = tf.layers.batch_normalization(hidden_layer_1, training=False)

'''

conv_layer_1 = tf.layers.conv2d(input_glimpses,
                                filters=32,
                                kernel_size=(3, 3),
                                strides=(1, 1),
                                padding='same',
                                activation=tf.nn.relu)

conv_layer_1_with_bn = tf.layers.batch_normalization(conv_layer_1, training=False)


conv_layer_2 = tf.layers.conv2d(conv_layer_1_with_bn,
                                filters=32,
                                kernel_size=(3, 3),
                                strides=(1, 1),
                                padding='same',
                                activation=tf.nn.relu)

conv_layer_2_with_bn = tf.layers.batch_normalization(conv_layer_2, training=False)
# final_conv_layer = normalized_image # change me

pool_layer_1 = tf.layers.max_pooling2d(conv_layer_2_with_bn,
                                       pool_size=(2, 2),
                                       strides=(2, 2))

conv_layer_3 = tf.layers.conv2d(pool_layer_1,
                                filters=64,
                                kernel_size=(3, 3),
                                strides=(1, 1),
                                padding='same',
                                activation=tf.nn.relu)

conv_layer_3_with_bn = tf.layers.batch_normalization(conv_layer_3, training=False)


conv_layer_4 = tf.layers.conv2d(conv_layer_3_with_bn,
                                filters=64,
                                kernel_size=(3, 3),
                                strides=(1, 1),
                                padding='same',
                                activation=tf.nn.relu)

conv_layer_4_with_bn = tf.layers.batch_normalization(conv_layer_4, training=False)

pool_layer_2 = tf.layers.max_pooling2d(conv_layer_4_with_bn,
                                       pool_size=(2, 2),
                                       strides=(2, 2))


#conv_layer_5 = tf.layers.conv2d(pool_layer_2,
#                                filters=128,
#                                kernel_size=(3, 3),
#                                strides=(1, 1),
#                                padding='same',
#                                activation=tf.nn.relu)

#conv_layer_5_with_bn = tf.layers.batch_normalization(conv_layer_5, training=True)


#conv_layer_6 = tf.layers.conv2d(conv_layer_5_with_bn,
#                                filters=128,
#                                kernel_size=(3, 3),
#                                strides=(1, 1),
#                                padding='same',
#                                activation=tf.nn.relu)

#conv_layer_6_with_bn = tf.layers.batch_normalization(conv_layer_6, training=True)

#pool_layer_3 = tf.layers.max_pooling2d(conv_layer_6_with_bn,
 #                                      pool_size=(4, 4),
  #                                     strides=(4, 4))




# convert 3d image to 1d tensor (don't change batch dimension)
flat_tensor = tf.contrib.layers.flatten(pool_layer_2)

# TODO improve fully connected layers
## Neural network hidden layers
hidden_layer_1 = tf.layers.dense(flat_tensor, 160, activation=tf.nn.relu)
hidden_layer_1_with_bn = tf.layers.batch_normalization(hidden_layer_1, training=False)
hidden_layer_2 = tf.layers.dense(hidden_layer_1_with_bn, 60, activation=tf.nn.relu)
hidden_layer_2_with_bn = tf.layers.batch_normalization(hidden_layer_2, training=False)

## Logit layer
logits = tf.layers.dense(hidden_layer_1_with_bn, 10)



summary_tensor = tf.summary.merge_all()

# Create list of variables except the ones we added
variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
variables.remove(feature_viz)
saver = tf.train.Saver(variables)

# use either loss function
## loss function to optimize over all the features in layer 2
# loss = -1 * tf.reduce_mean(conv_layer_2)

## Loss function to optimize for a single feature (#5) in layer 1
feature_mask = tf.constant([[
    [1 if feature == 5 else 0
     for feature in range(32)]
    for y in range(32)]
    for x in range(32)],
    tf.float32)
loss = -1 * tf.reduce_mean(conv_layer_1 * feature_mask)

## backpropagation algorithm
# only update the weights of the feature viz image
train = tf.train.AdamOptimizer(0.5).minimize(loss, var_list=[feature_viz])




## Make tensorflow session
with tf.Session() as sess:
    viz_summary_writer = tf.summary.FileWriter(TENSORBOARD_LOGDIR + "/viz", sess.graph)

    ## Initialize variables
    sess.run(tf.global_variables_initializer())

    # Restore saved model
    saver.restore(sess, "model/model.ckpt")

    step_count = 0
    while True:
        step_count += 1

        print("step {}".format(step_count))

        # train network
        glimpses, batch_number = createRandomGlimpses(50)

        _ = sess.run([train], feed_dict={glipmse_offset_placeholder: glimpses, glipmse_batch_number_placeholder: batch_number})

        # every 10 steps post new image
        if step_count % 10 ==  0:
            summary = sess.run([summary_tensor])[0]

            # write data to tensorboard
            viz_summary_writer.add_summary(summary, step_count)

        # stop training after 1,000 steps
        if step_count > 3000:
            break