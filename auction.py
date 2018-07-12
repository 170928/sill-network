import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import xavier_initializer





class Model():
    def __init__(self):
        self.Nuser = 5
        self.Ngroup = 5
        self.Nunit = 10

        self.X = tf.placeholder(dtype=tf.float32, shape=[self.Nuser], name='Input')

        self.split1, self.split2, self.split3, self.split4, self.split5 = tf.split(self.X, num_or_size_splits=5, axis=0)

        self.split1 = tf.Print(self.split1, [self.split1], 'bid 1 :: ')
        self.split2 = tf.Print(self.split2, [self.split2], 'bid 2 :: ')
        self.split3 = tf.Print(self.split3, [self.split3], 'bid 3 :: ')
        self.split4 = tf.Print(self.split4, [self.split4], 'bid 4 :: ')
        self.split5 = tf.Print(self.split5, [self.split5], 'bid 5 :: ')

        self.t1 = [self.layer('l1', self.split1)]
        self.t2 = [self.layer('l2', self.split2)]
        self.t3 = [self.layer('l3', self.split3)]
        self.t4 = [self.layer('l4', self.split4)]
        self.t5 = [self.layer('l5', self.split5)]

        #transformed_bid 는 (5, ) 의 tensor
        self.transformed_bid = tf.concat([self.t1, self.t2, self.t3, self.t4, self.t5], 0)
        #print(self.transformed_bid)


    def layer(self, name, x):

        with tf.variable_scope( name ):
            weight = [tf.get_variable(name + "w" + str(i), [1], initializer = xavier_initializer()) for i in range(self.Ngroup * self.Nunit)]
            bias = [tf.get_variable(name + "b" + str(i), [1]) for i in range(self.Ngroup * self.Nunit)]
            result = [tf.add(tf.multiply(weight[i], x), bias[i])for i in range(self.Ngroup*self.Nunit)]
            #result는 50개의 (1,) 짜리 tensor의 배열
            result = tf.concat(result,0)
            #result는 (50, )의 tensor가 되었다
            result = tf.reshape(result, shape = [self.Ngroup, self.Nunit])
            #reulst는 (5, 10)의 tensor가 되었다
            max_out = tf.reduce_max(result, axis=1)
            min_out = tf.reduce_min(max_out, axis=0)
            return min_out






def main():
    model = Model()

    input = np.array([5,2,3,4,1])
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("./logs/tmp",sess.graph)

        sess.run( model.transformed_bid , feed_dict={model.X: input})
        #writer.add_summary(summary)


if __name__ == "__main__":
    main()