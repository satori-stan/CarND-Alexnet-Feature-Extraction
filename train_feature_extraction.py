import numpy as np
import pickle
import tensorflow as tf
from alexnet import AlexNet
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# TODO: Load traffic signs data.
with open('train.p', mode='rb') as f:
    train = pickle.load(f)

# TODO: Split data into training and validation sets.
X_train, X_valid, y_train, y_valid = train_test_split(train.features, train.labels)

# TODO: Define placeholders and resize operation.
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
resized = tf.image.resize_images(x, (227, 227))
y = tf.placeholder(tf.int32, (None))

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)

# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
#sign_names = pd.read_csv('signnames.csv') # Could come from the sign names
nb_classes = len(np.unique(train.labels)) # 43
shape = (fc7.get_shape().as_list()[-1], nb_classes)  # use this shape for the weight matrix
print(shape)
fc8W = tf.Variable(tf.truncated_normal(shape))
fc8b = tf.Variable(tf.truncated_normal((shape[-1],)))
#probs = tf.nn.softmax(tf.matmul(fc7, fc8W) + fc8b)
logits = tf.matmul(fc7, fc8W) + fc8b

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
rate = 0.001
EPOCHS = 3
BATCH_SIZE = 64

one_hot_y = tf.one_hot(y, nb_classes)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# TODO: Train and evaluate the feature extraction model.
def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy, prediction = sess.run((accuracy_operation, logits), feed_dict={x: batch_x, y: batch_y, keep_prob: 1})
        total_accuracy += (accuracy * len(batch_x))
    return (total_accuracy / num_examples, prediction)

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            _,minimized_loss = sess.run((training_operation, loss_operation), feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
            
        validation_accuracy,_ = evaluate(X_valid, y_valid)
        print("EPOCH {} ... Training loss = {:.5f}, Validation Accuracy = {:.3f}".format(i+1, minimized_loss, validation_accuracy))
        
    saver.save(sess, './alexnet_fe')
    print("Model saved")

