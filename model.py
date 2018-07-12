import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
import numpy as np
from random import *

class model:

    def __init__(self, in_dim, trainphase):

        self.in_dim = in_dim
        self.group_dim = 5
        self.group_unit = 10
        self.trainphase = trainphase

        # in_dim 는 bid N개를 의미한다
        self.X = tf.placeholder(tf.float32, [in_dim])

        #1x5 tensor
        self.transformed_bid = self.transform()


        #1x5 tensor allocation rule
        self.allocation = self.softmax(tf.constant(2, dtype=tf.float32))
        #print(self.allocation)

        #1x5 paymnet rule
        self.payment = self.payment()
        print(self.payment)

    def payment(self):
        partition = tf.convert_to_tensor(np.arange(0, self.in_dim), dtype=tf.int32)
        partition = tf.one_hot(partition, self.in_dim)
        partition = tf.to_int32(partition)


        for i in range(0, self.in_dim):
            temp, _ = tf.dynamic_partition(self.transformed_bid, partitions=[partition[i, :]], num_partitions=2)
            if i == 0:
                result = [[tf.nn.relu(tf.reduce_max(temp, 0))]]
            else:
                result = tf.concat([result, [[tf.nn.relu(tf.reduce_max(temp, 0))]]], 1)
        return result

    def softmax(self, k):
        # sum will be 1x1 tensor
        self.sum = tf.add(tf.reduce_sum(tf.exp( tf.multiply(k, self.transformed_bid)), axis=1, keepdims=True), tf.constant(1, tf.float32))
        # out will be 1xN tensor
        self.out = tf.div(tf.exp(tf.multiply(k, self.transformed_bid)) , self.sum)
        return self.out

    def transform(self):
        for idx in range(0, self.in_dim):
            if idx == 0:
                self.output = self.forward(tf.slice(self.X, [idx] ,[1]), idx)
            else:
                self.output = tf.concat([self.output, self.forward(tf.slice(self.X, [idx] ,[1]), idx)], 1)
        return self.output

    def nn(self, x, name):
        with tf.name_scope( name ) as scope:
            w = tf.get_variable(name= name + 'w', shape = [1,1], initializer=xavier_initializer())
            b = tf.get_variable(name= name + 'b', shape = [1,1], initializer=xavier_initializer())

            #tf.summary.histogram('Weight', w)
            #tf.summary.histogram('Bias', b)

            return tf.add(tf.multiply(w,x), b)

    def forward(self, x, idx):

        #for문으로 self.nn을 K group번 J unit번 돌리면 되지않을까?

        for i in range(0, self.group_dim):
            for j in range(0, self.group_unit):
                if j == 0:
                    self.hidden_out = self.nn( x, str(idx) + "h"+str(i)+"u"+str(j))
                else:
                    self.hidden_out = tf.concat([self.hidden_out, self.nn(x, str(idx) + "h"+str(i)+"u"+str(j))], 1)

            if i == 0:
                self.group_out = [tf.reduce_max(self.hidden_out, reduction_indices=[1])]
            else:
                self.group_out = tf.concat([self.group_out, [tf.reduce_max(self.hidden_out, reduction_indices=[1])]], 1)

        return [tf.reduce_min(self.group_out, reduction_indices=[1])]

if __name__ == "__main__":

    model = model(5,True)


    with tf.name_scope("cost") as scope:
        sum = tf.matmul(model.transformed_bid, tf.transpose(model.allocation))

        cost =  tf.multiply(tf.to_float(-1),tf.squeeze(sum))
        c = tf.summary.scalar('cost', cost)

    with tf.name_scope("train") as scope:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.002).minimize(cost)

    init = tf.global_variables_initializer()


    with tf.Session() as sess:

        sess.run(init)

        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("./logs/tmp",sess.graph)

        for step in range(10):
            input = np.array([1,10,5,7,2])
            co, summary, _ = sess.run( [cost, merged, optimizer] , feed_dict = {model.X: input})
            print(co)
            writer.add_summary(summary, step)