import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

'''
input > weight > hidden layer 1 (activation function) 
      > weights > hidden layer 2 (activation function)
      > weights > output layer

compare output to intended output > cost or loss function (cross entropy)
optimization function(optimizer) > minimize cost (AdamOptimizer... SGD, AdaGrad)

backpropagation

feed forward + backprop = epoch
'''

mnist = input_data.read_data_sets("./data", one_hot=True)

'''
10 个分类, 0-9
one_hot 指的是向量中只有一个被置成 1
0 = [1, 0, 0, 0, 0, 0, 0, 0, 0]
1 = [0, 1, 0, 0, 0, 0, 0, 0, 0]
    .
    .
9 = [0, 0, 0, 0, 0, 0, 0, 0, 1]
'''

n_nods_hl1 = 500
n_nods_hl2 = 500
n_nods_hl3 = 500

n_classes = 10
batch_size = 100

# height * width
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def neural_network_model(data):
    
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nods_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nods_hl1]))}
    
    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nods_hl1, n_nods_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nods_hl2]))}
    
    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nods_hl2, n_nods_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nods_hl3]))}
    
    output_layer = {'weights':tf.Variable(tf.random_normal([n_nods_hl3, n_classes])),
                      'biases':tf.Variable(tf.random_normal([n_classes]))}

    # (input_data * weight) +biase
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1) # activation function

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2) # activation function

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3) # activation function

    output = tf.add(tf.matmul(l2, output_layer['weights']), output_layer['biases'])
    return output

def trian_neural_network(x, y):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # cycles feed forward + backprop
    hm_epochs = 10

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x:epoch_x, y:epoch_y})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)
        
        correct = tf.equal(tf.math.argmax(prediction, 1), tf.math.argmax(y, 1))
        
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        print('Accuracy:', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

trian_neural_network(x, y)