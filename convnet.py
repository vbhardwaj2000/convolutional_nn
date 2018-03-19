"""
#TODO change this comment to something describing your project
Project 2

At the end you should see something like this
Step Count:300
Training accuracy: 0.880000 loss: 0.444277
Test accuracy: 0.620000 loss: 1.418351

play around with your model to try and get an even better score
"""

import tensorflow as tf
import dataUtils

TENSORBOARD_LOGDIR = "logdir"

# Clear the old log files
dataUtils.deleteDirectory(TENSORBOARD_LOGDIR)

### Build tensorflow blueprint ###
# Tensorflow placeholder
input_placeholder = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])

# View sample inputs in tensorboard
tf.summary.image("input_image", input_placeholder)

# Normalize image
# Subtract off the mean and divide by the variance of the pixels.

random_image = tf.map_fn(lambda frame: tf.image.random_flip_left_right(frame),input_placeholder)
tf.size(random_image, name=None, out_type=tf.int32)
#tf.size(random_image,dtypes.int32)
normalized_image = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), random_image)

# TODO add conv layers here

conv_layer_1 = tf.layers.conv2d(normalized_image,
                                filters=32,
                                kernel_size=(3, 3),
                                strides=(1, 1),
                                padding='same',
                                activation=tf.nn.relu)

conv_layer_1_with_bn = tf.layers.batch_normalization(conv_layer_1, training=True)


conv_layer_2 = tf.layers.conv2d(conv_layer_1_with_bn,
                                filters=32,
                                kernel_size=(3, 3),
                                strides=(1, 1),
                                padding='same',
                                activation=tf.nn.relu)

conv_layer_2_with_bn = tf.layers.batch_normalization(conv_layer_2, training=True)
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

conv_layer_3_with_bn = tf.layers.batch_normalization(conv_layer_3, training=True)


conv_layer_4 = tf.layers.conv2d(conv_layer_3_with_bn,
                                filters=64,
                                kernel_size=(3, 3),
                                strides=(1, 1),
                                padding='same',
                                activation=tf.nn.relu)

conv_layer_4_with_bn = tf.layers.batch_normalization(conv_layer_4, training=True)

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
hidden_layer_1_with_bn = tf.layers.batch_normalization(hidden_layer_1, training=True)
hidden_layer_2 = tf.layers.dense(hidden_layer_1_with_bn, 60, activation=tf.nn.relu)
hidden_layer_2_with_bn = tf.layers.batch_normalization(hidden_layer_2, training=True)

## Logit layer
logits = tf.layers.dense(hidden_layer_1_with_bn, 10)

# label placeholder
label_placeholder = tf.placeholder(tf.uint8, shape=[None])
label_one_hot = tf.one_hot(label_placeholder, 10)

# loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=label_one_hot, logits=logits))

# TODO choose better backpropagation
# backpropagation algorithm
#train = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
train = tf.train.AdamOptimizer().minimize(loss)

accuracy = dataUtils.accuracy(logits, label_one_hot)

# summaries
tf.summary.scalar('accuracy', accuracy)
tf.summary.scalar('loss', loss)

tf.summary.tensor_summary("logits", logits)
tf.summary.tensor_summary("labels", label_one_hot)
summary_tensor = tf.summary.merge_all()

saver = tf.train.Saver()

## Make tensorflow session
with tf.Session() as sess:
    training_summary_writer = tf.summary.FileWriter(TENSORBOARD_LOGDIR + "/training", sess.graph)
    test_summary_writer = tf.summary.FileWriter(TENSORBOARD_LOGDIR + "/test", sess.graph)

    ## Initialize variables
    sess.run(tf.global_variables_initializer())

    step_count = 0
    while True:
        step_count += 1

        # get batch of training data
        batch_training_data, batch_training_labels = dataUtils.getCIFAR10Batch(is_eval=False, batch_size=100)

        # train network
        training_accuracy, training_loss, summary, _ = sess.run([accuracy, loss, summary_tensor, train],
                                                                feed_dict={input_placeholder: batch_training_data,
                                                                           label_placeholder: batch_training_labels})

        # write data to tensorboard
        training_summary_writer.add_summary(summary, step_count)

        # every 10 steps check accuracy
        if step_count % 10 == 0:
            # get Batch of test data
            batch_test_data, batch_test_labels = dataUtils.getCIFAR10Batch(is_eval=True, batch_size=100)

            # do eval step to test accuracy
            test_accuracy, test_loss, summary = sess.run([accuracy, loss, summary_tensor],
                                                         feed_dict={input_placeholder: batch_test_data,
                                                                    label_placeholder: batch_test_labels})

            # write data to tensorboard
            test_summary_writer.add_summary(summary, step_count)

            print("Step Count:{}".format(step_count))
            print("Training accuracy: {:.6f} loss: {:.6f}".format(training_accuracy, training_loss))
            print("Test accuracy: {:.6f} loss: {:.6f}".format(test_accuracy, test_loss))

        if step_count % 100 == 0:
            save_path = saver.save(sess, "model/model.ckpt")

        #print('----stepcount is {}',step_count)
        # stop training after 1,000 steps-
        if step_count > 1000:
            break
