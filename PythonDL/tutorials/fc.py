import tensorflow as tf  # 导入TensorFlow库（1.x版本）

class FCLayer():
    def __init__(self,input):
        self.x = input  # 输入张量形状 [1,2]
        self.initialize()
        
    def initialize(self):
        # 定义两层全连接层的权重、偏置
        self.w1 = tf.Variable(tf.constant([[0.2,0.1,0.3],[0.2,0.4,0.3]],shape=[2,3]),name='w1')
        self.b1 = tf.constant([-0.3,0.1,0.2],shape=[1,3],name='b1')
        self.w2 = tf.Variable(tf.constant([0.2,0.5,0.25],shape=[3,1]),name='w2')
        self.b2 = tf.constant([-0.3],shape=[1],name='b2')
        # 前向传播计算逻辑
        hidden_layer = tf.matmul(self.x,self.w1)+self.b1
        self.y = tf.matmul(hidden_layer,self.w2)+self.b2

    def calculate_and_print(self):
        init_op = tf.global_variables_initializer()  # TensorFlow 1.x 中，Variable需要手动初始化，这个操作会初始化所有定义的Variable（这里是w1和w2）
        with tf.Session() as sess:
            sess.run(init_op)
            result = sess.run(self.y)
            print('fc result is: %.3f' % result)


if __name__ == '__main__':
    input = tf.constant([0.9,0.85],shape=[1,2])
    FCLayer = FCLayer(input)
    FCLayer.calculate_and_print()


# TensorFlow 2.x 等价实现
import tensorflow as tf

# 定义两层全连接层（用Keras高级API）
model = tf.keras.Sequential([
    tf.keras.layers.Dense(3, input_shape=(2,),
                          kernel_initializer=tf.constant_initializer([[0.2,0.1,0.3],[0.2,0.4,0.3]]),
                          bias_initializer=tf.constant_initializer([-0.3,0.1,0.2])),
    tf.keras.layers.Dense(1,
                          kernel_initializer=tf.constant_initializer([[0.2],[0.5],[0.25]]),
                          bias_initializer=tf.constant_initializer([-0.3]))
])

# 输入数据
input_data = tf.constant([[0.9, 0.85]])
# 前向传播（2.x动态图，直接调用即可）
result = model(input_data)
print(f'fc result is: {result.numpy()[0][0]:.3f}')