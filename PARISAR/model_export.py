from __future__ import division, print_function, absolute_import
# library for optmising inference
from tensorflow.python.tools import optimize_for_inference_lib, freeze_graph
import tensorflow as tf
# Higher level API tflearn
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import numpy as np
import cv2
import glob
 # Data loading and preprocessing
#helper functions to download the CIFAR 10 data and load them dynamically
#from tflearn.datasets import cifar10
import matplotlib.pyplot as plt
#####################
#grab training images
#####################

X_data = []
Y_data =[]
classes = ['defense', 'eiffel', 'invalides', 'louvre', 'moulinrouge', 'museedorsay', 'notredame', 'pantheon', 'pompidou', 'sacrecoeur', 'triomphe']
for x in classes:
    files = []
    files = glob.glob ("C://Users//cosmi//Desktop//train/train/{}/*.jpg".format(x))
    print(x)
    print(files)
    for myFile in range(len(files)):
        print(str(myFile) + 'train')
        image = cv2.resize(cv2.imread(files[myFile]),(32,32))
        X_data.append(image)
        Y_data.append(classes.index(x))
        if myFile == 49:
            break
#################
#grab test images
#################
number_of_train_images = 50
number_of_test_images  = 20
Xtest_data =[]
Ytest_data =[]
for x in classes:
    files = []
    files = glob.glob ("C://Users//cosmi//Desktop//train/train/{}/*.jpg".format(x))
    for myFile in range(len(files)):
        print(str(myFile+number_of_train_images-1) + 'test' + x)
        image = cv2.resize(cv2.imread(files[myFile+number_of_train_images]),(32,32))
        Xtest_data.append(image)
        Ytest_data.append(classes.index(x))
        if myFile == number_of_test_images-1:
            break
##################################################################



X, Y = shuffle(X_data, Y_data)
Y = to_categorical(Y,11)
Y_test = to_categorical(Ytest_data,11)

print('Y_data labels:{}'.format(Y_data))
print('X_data shape:', np.array(X_data).shape)
print('Ytest_data labels:{}'.format(Ytest_data))
print('Categorical Y_test:{}'.format(Y_test[0:20]))
print('Xtest_data shape:', np.array(Xtest_data).shape)


plt.interactive(False)
plt.figure, plt.imshow(X_data[0])
plt.show(block = True)

 #input image
x=tf.placeholder(tf.float32,shape=[None, 32, 32, 3] , name='ipnode1')
#input class
y_=tf.placeholder(tf.float32,shape=[None, 11] , name='input_class')
 # AlexNet architecture
input_layer=x
network = conv_2d(input_layer, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = fully_connected(network, 512, activation='relu')
network = fully_connected(network, 11, activation='linear')
y_predicted=tf.nn.softmax(network , name="opnode")
 #loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_predicted+np.exp(-10)), reduction_indices=[1]))
#optimiser -
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#calculating accuracy of our model
correct_prediction = tf.equal(tf.argmax(y_predicted,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
 #TensorFlow session
config = tf.ConfigProto(device_count = {'GPU':0})
#sess = tf.Session(config=config)
sess = tf.Session()
#initialising variables
init = tf.global_variables_initializer()
sess.run(init)
#tensorboard for better visualisation
writer =tf.summary.FileWriter('tensorboard/', sess.graph)
epoch=1 # run for more iterations according your hardware's power
#change batch size according to your hardware's power. For GPU's use batch size in powers of 2 like 2,4,8,16...
batch_size=32
no_itr_per_epoch=len(X)//batch_size
n_test=len(Xtest_data) #number of test samples
 # Commencing training process
for iteration in range(epoch):
    print("Iteration no: {} ".format(iteration))
    previous_batch=0
    # Do our mini batches:
    for i in range(no_itr_per_epoch):
        current_batch=previous_batch+batch_size
        x_input=X[previous_batch:current_batch]
        x_images=np.reshape(x_input,[batch_size,32,32,3])
        y_input=Y[previous_batch:current_batch]
        y_label=np.reshape(y_input,[batch_size,11])
        previous_batch=previous_batch+batch_size
        #_,loss=sess.run([train_step, cross_entropy], feed_dict={x: x_images,y_: y_label})
        #if i % 100==0 :
        #    print ("Training loss : {}" .format(loss))
    x_test_images=np.reshape(Xtest_data[0:n_test],[n_test,32,32,3])
    y_test_labels=np.reshape(Y_test[0:n_test],[n_test,11])
    Accuracy_test=sess.run(accuracy,
                           feed_dict={
                        x: x_test_images ,
                        y_: y_test_labels
                      })
    # Accuracy of the test set
    Accuracy_test=round(Accuracy_test*100,2)
    print("Accuracy ::  Test_set {} %  " .format(Accuracy_test))
saver = tf.train.Saver()
model_directory='model_files/'
#saving the graph
tf.train.write_graph(sess.graph_def, model_directory, 'savegraph.pbtxt')
saver.save(sess, 'model_files/model.ckpt')
# Freeze the graph
MODEL_NAME = 'CIFARlaptop'
input_graph_path = 'model_files/savegraph.pbtxt'
checkpoint_path = 'model_files/model.ckpt'
input_saver_def_path = ""
input_binary = False
output_node_names = "opnode"
restore_op_name = "save/restore_all"
filename_tensor_name = "save/Const:0"
output_frozen_graph_name = 'model_files/frozen_model_'+MODEL_NAME+'.pb'
output_optimized_graph_name = 'model_files/optimized_inference_model_'+MODEL_NAME+'.pb'
clear_devices = True
#Freezing the graph and generating protobuf files
freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                          input_binary, checkpoint_path, output_node_names,
                          restore_op_name, filename_tensor_name,
                          output_frozen_graph_name, clear_devices, "")
#Optimising model for inference only purpose
output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        sess.graph_def,
        ["ipnode1"], # an array of the input node(s)
        ["opnode"], # an array of output nodes
        tf.float32.as_datatype_enum)
with tf.gfile.GFile(output_optimized_graph_name, "wb") as f:
            f.write(output_graph_def.SerializeToString())

tf.global_variables()


graph = tf.get_default_graph()
list_of_tuples = [op.values() for op in graph.get_operations()]

print(list_of_tuples)
sess.close()
