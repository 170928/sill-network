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

        #self.split1 = tf.Print(self.split1, [self.split1], 'bid 1 :: ')
        #self.split2 = tf.Print(self.split2, [self.split2], 'bid 2 :: ')
        #self.split3 = tf.Print(self.split3, [self.split3], 'bid 3 :: ')
        #self.split4 = tf.Print(self.split4, [self.split4], 'bid 4 :: ')
        #self.split5 = tf.Print(self.split5, [self.split5], 'bid 5 :: ')

        with tf.variable_scope('Transforming'):

            self.t1 = [self.layer('l1', self.split1)]
            self.t2 = [self.layer('l2', self.split2)]
            self.t3 = [self.layer('l3', self.split3)]
            self.t4 = [self.layer('l4', self.split4)]
            self.t5 = [self.layer('l5', self.split5)]

            #transformed_bid 는 (5, ) 의 tensor
            self.transformed_bid = tf.concat([self.t1, self.t2, self.t3, self.t4, self.t5], 0)
            self.transformed_bid = tf.Print(self.transformed_bid, [self.transformed_bid], "Transformed bid :: ")
            #print(self.transformed_bid)
            self.sp1, self.sp2, self.sp3, self.sp4, self.sp5 = tf.split(self.transformed_bid, num_or_size_splits=5, axis=0)


        with tf.variable_scope('Allocation'):
            # Softmax에 사용될 분모 :: sum will be (1, ) tensor
            k = tf.to_float(2)
            self.sum = tf.add(tf.reduce_sum(tf.exp( tf.multiply(k, self.transformed_bid) ), axis = 0, keepdims=True), tf.constant(1, tf.float32))
            self.sum = tf.Print(self.sum, [self.sum], "Alloc Sum :: ")

            self.alloc1 = self.allocation(k, self.sp1)
            self.alloc2 = self.allocation(k, self.sp2)
            self.alloc3 = self.allocation(k, self.sp3)
            self.alloc4 = self.allocation(k, self.sp4)
            self.alloc5 = self.allocation(k, self.sp5)

            self.allocation_prob = tf.concat([self.alloc1, self.alloc2, self.alloc3, self.alloc4, self.alloc5 ], axis = 0)

        with tf.variable_scope('Payment'):

            self.pay1 = [self.payment(tf.concat([self.sp2, self.sp3, self.sp4, self.sp5, [0]], axis = 0))]
            self.pay2 = [self.payment(tf.concat([self.sp1, self.sp3, self.sp4, self.sp5, [0]], axis = 0))]
            self.pay3 = [self.payment(tf.concat([self.sp1, self.sp2, self.sp4, self.sp5, [0]], axis = 0))]
            self.pay4 = [self.payment(tf.concat([self.sp1, self.sp2, self.sp3, self.sp5, [0]], axis = 0))]
            self.pay5 = [self.payment(tf.concat([self.sp1, self.sp2, self.sp3, self.sp4, [0]], axis = 0))]

            self.payment = tf.concat([self.pay1, self.pay2, self.pay3, self.pay4, self.pay5], axis = 0)
            self.payment = tf.Print(self.payment, [self.payment], "Payment is ::")
            #print(self.payment)

        with tf.variable_scope("inverse"):
            self.variables = tf.trainable_variables()
            #variable은 vector로 우리가 꺼내서 사용가능 weight 0~49 bias 50~99 까지 할당 (100, ) vector이다.
            #print(self.variables[:100])

            self.p1 = [self.inverse(self.pay1, 'l1' , 0)]
            self.p2 = [self.inverse(self.pay2, 'l2' , 1)]
            self.p3 = [self.inverse(self.pay3, 'l3' , 2)]
            self.p4 = [self.inverse(self.pay4, 'l4' , 3)]
            self.p5 = [self.inverse(self.pay5, 'l5' , 4)]

            self.result = tf.concat([self.p1, self.p2, self.p3, self.p4, self.p5], axis = 0)
            #print(self.result)


    def inverse(self, x, name, layer):
        x = tf.Print(x,[x], name + "  Payment is :: " )
        #temp는 input x로 이루어진 (50, ) vecotr
        temp = tf.tile(x, [self.Ngroup*self.Nunit])

        bias = tf.concat(self.variables[50*(2*layer+1) : 50*(2*(layer+1))], axis=0)
        weight = tf.concat(self.variables[50*(2*layer)  : 50*(2*layer+1)], axis=0)
        #weight = tf.Print(weight, [weight], "Weigh set ::: ")

        #print(temp, bias, weight)
        #bias = tf.Print(bias, [bias], "Bias //// ")

        bias_out = tf.subtract(temp, bias)
        #bias_out = tf.Print(bias_out, [bias_out],"Bias Out ::: ")
        weight_out = tf.divide(bias_out, weight)
        #weight_out = tf.Print(weight_out, [weight_out], "Weight Out ::: ")

        result = tf.reshape(weight_out, shape=[self.Ngroup, self.Nunit])
        min_out = tf.reduce_min(result, axis=1)
        max_out = tf.reduce_min(min_out, axis=0)

        return max_out

    def payment(self, x):
        #print(x)
        #x = tf.Print(x,[x], "Input :: ")
        out = tf.reduce_max(x, axis=0)
        #out = tf.Print(out, [out], "Max is :: ")
        #print(out)

        return tf.nn.relu(out)

    def allocation(self, k, x):
        # out will be 1xN tensor
        out = tf.div(tf.exp(tf.multiply(k, x)) , self.sum)
        return out





    def layer(self, name, x):

        with tf.variable_scope( name ):

            weight = [ tf.clip_by_value(tf.get_variable( "w" + str(i), [1], initializer = xavier_initializer()), 0.01, np.infty) for i in range(self.Ngroup * self.Nunit) ]
            bias = [ tf.clip_by_value(tf.get_variable( "b" + str(i), [1], initializer = xavier_initializer()), 0.01, np.infty) for i in range(self.Ngroup * self.Nunit) ]
            result = [ tf.add(tf.multiply(weight[i], x), bias[i])for i in range(self.Ngroup*self.Nunit) ]
            #result는 50개의 (1,) 짜리 tensor의 배열
            result = tf.Print(result, [result], "Output :: ")
            result = tf.concat(result,0)
            #result는 (50, )의 tensor가 되었다
            result = tf.reshape(result, shape = [self.Ngroup, self.Nunit])
            #reulst는 (5, 10)의 tensor가 되었다
            max_out = tf.reduce_max(result, axis=1)
            #max_out = tf.Print(max_out,[max_out], "Max out :: ")
            min_out = tf.reduce_min(max_out, axis=0)
            #min_out = tf.Print(min_out, [min_out], "Min out :: ")
            return min_out






def main():
    model = Model()

    #승자 index
    winnerIdx = tf.argmax(model.allocation_prob)
    allocation = model.allocation_prob
    payment = model.result

    with tf.name_scope("cost") as scope:
        mul1 = tf.multiply(payment, allocation)
        mul2 = tf.multiply(tf.to_float(-1), mul1)
        cost = tf.reduce_sum(mul2, axis=0)

    with tf.name_scope("train") as scope:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)



    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("./logs/tmp",sess.graph)

        for i in range(2):
            #print("Step [", i, "]")
            input = np.random.randint(0,50,(5,))
            #print(input, input.shape)


            c, _, win, pay, alloc = sess.run( [cost, optimizer, winnerIdx, payment, allocation], feed_dict={model.X: input})
            #writer.add_summary(summary)
            if i%100 == 0:
                print( "Input :: ", input, "Winner Idx :: ", win, "Payment :: ", pay[win], "Cost :: ", c, "Allocation Prob ::", alloc )

if __name__ == "__main__":
    main()