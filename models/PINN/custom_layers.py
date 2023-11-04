import tensorflow as tf
from tensorflow.keras.initializers import Constant
from tensorflow.keras.constraints import MinMaxNorm
from tensorflow.keras.constraints import Constraint


class ScalarMinMaxConstraint(Constraint):
    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, w):
        return tf.clip_by_value(w, self.min_value, self.max_value)

    def get_config(self):
        return {'min_value': self.min_value, 'max_value': self.max_value}


# 1. 自定义IDM模型的层
class IDM_Layer(tf.keras.layers.Layer):
    def __init__(self, forward_steps, **kwargs):
        self.forward_steps = forward_steps
        super(IDM_Layer, self).__init__(**kwargs)

    def build(self, input_shape):
        # 初始化IDM参数
        self.vf = self.add_weight(name='vf',
                                  shape=(),
                                  initializer=Constant(25),
                                  trainable=True,
                                  constraint=ScalarMinMaxConstraint(0., 30.))

        self.A = self.add_weight(name='A',
                                 shape=(),
                                 initializer=Constant(2),
                                 trainable=True,
                                 constraint=ScalarMinMaxConstraint(0., 5.))

        self.b = self.add_weight(name='b',
                                 shape=(),
                                 initializer=Constant(3),
                                 trainable=True,
                                 constraint=ScalarMinMaxConstraint(0., 5.))

        self.s0 = self.add_weight(name='s0',
                                  shape=(),
                                  initializer=Constant(3),
                                  trainable=True,
                                  constraint=ScalarMinMaxConstraint(0., 10.))

        self.T = self.add_weight(name='T',
                                 shape=(),
                                 initializer=Constant(2),
                                 trainable=True,
                                 constraint=ScalarMinMaxConstraint(1., 3.))

        super(IDM_Layer, self).build(input_shape)

    def call(self, inputs):
        vi, delta_v, delta_d = inputs
        predictions = []

        for _ in range(self.forward_steps):
            s_star = self.s0 + tf.maximum(tf.constant(0.0, dtype=tf.float32), vi*self.T + (vi * delta_v) / (2 * (self.A*self.b) ** 0.5))
            epsilon = 1e-5
            ahat = self.A * (1 - (vi/self.vf)**4 - (s_star / (delta_d + epsilon)) ** 2)
            predictions.append(ahat)

            # 更新为下一个时间步的输入
            vi = vi + ahat*0.1
            delta_v = delta_v - ahat*0.1

        return tf.stack(predictions, axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "forward_steps": self.forward_steps
        })
        return config