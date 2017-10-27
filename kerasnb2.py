import tensorflow as tf
sess = tf.Session()
import tensorlayer as tl
from tensorlayer.iterate import minibatches
from keras import backend as K
K.set_session(sess)
from keras.models import  Model
from keras.layers import Dense,Dropout,Input
batch_size = 128
img = tf.placeholder(tf.float32, shape=(None, 784))
y_ = tf.placeholder(tf.int64, shape=(None, ))

goon = False
# Keras layers can be called on TensorFlow tensors:
inputs = Input(shape=(784,))
x = Dense(128, activation='relu')(inputs)  # fully-connected layer with 128 units and ReLU activation
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
x_out = Dense(10, activation='linear')(x)
x_model = Model(input=inputs,output=x_out)
y = x_model(img)

#mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)
X_train, y_train, X_val, y_val, X_test, y_test = \
                tl.files.load_mnist_dataset(shape=(-1, 784))
cost = tl.cost.cross_entropy(y, y_, 'cost')
correct_prediction = tf.equal(tf.argmax(y, 1), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
n_epoch = 200
learning_rate = 0.0001

# Initialize all variables
train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999,
    epsilon=1e-08, use_locking=False).minimize(cost)
init_op = tf.global_variables_initializer()
sess.run(init_op)
# Run training loop
for layer in x_model.layers:
    layer.trainable = False
with sess.as_default():
    #saver = tf.train.Saver()
    if goon:
        #saver.restore(sess,'./model.ckpt')
        x_model.load_weights('x_model.h5')
        print('OK load!!!!!!!!!!')
    else:
        print('Creat new!!!!!!!!!!')
    

    
    for i in range(100):
        for X_train_a, y_train_a in minibatches(
                                X_train, y_train, batch_size, shuffle=True):
            _, _ = sess.run([cost, train_op], feed_dict={img: X_train_a, y_: y_train_a,
                                K.learning_phase(): 1})
        
        train_loss, train_acc, n_batch = 0, 0, 0
        for X_train_a, y_train_a in minibatches(
                            X_train, y_train, batch_size, shuffle=False):
            err, ac = sess.run([cost, acc], feed_dict={img: X_train_a, y_: y_train_a,
                                K.learning_phase(): 0})
            train_loss += err; train_acc += ac; n_batch += 1
        if (i+1)%5==0:
            #saver.save(sess, "./model.ckpt")
            x_model.save_weights('x_model.h5')
            print('save model!!!!!!')
        print("   train loss: %f" % (train_loss/ n_batch))
        print("   train acc: %f" % (train_acc/ n_batch))
        val_loss, val_acc, n_batch = 0, 0, 0
        for X_val_a, y_val_a in minibatches(
                                X_val, y_val, batch_size, shuffle=False):
            err, ac = sess.run([cost, acc], feed_dict={img: X_val_a, y_: y_val_a,
                                K.learning_phase(): 0})
            val_loss += err; val_acc += ac; n_batch += 1
        print("   val loss: %f" % (val_loss/ n_batch))
        print("   val acc: %f" % (val_acc/ n_batch))