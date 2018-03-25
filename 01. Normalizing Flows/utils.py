import numpy as np
import tensorflow as tf
import gpflow as gp
from gpflow import transforms
from tensorflow.python.framework import random_seed
import numpy
import matplotlib.pyplot as plt
from tensorflow import Variable
float_type = tf.float32
jitter_level = 1e-4


class Variable(Variable):
    '''
    extend tf.Variable class to have an additional property : learning_rate
    '''
    pass

    def set_learning_rate(self,value):
        self._learning_rate = value

    @property
    def learning_rate(self):
        if hasattr(self,'_learning_rate'):
            return self._learning_rate

        else:
            return 0.001

class Param:
    '''
    Copied from GPFlow https://github.com/GPflow/GPflow/tree/master/gpflow/params
    '''
    def __init__(self,value,transform = None,fixed=False,name=None,learning_rate=None,summ=False):
        self.value = value
        self.fixed = fixed

        if name is None:
            self.name = "param"
        else:
            self.name = name

        if transform is None:
            self.transform=transforms.Identity()
        else:
            self.transform = transform

        if self.fixed:
            self.tf_opt_var = tf.constant(self.value,name=name,dtype=float_type)
        else:
            self.tf_opt_var = Variable(self.transform.backward(self.value),name=name,dtype=float_type)

        if learning_rate is not None:
            self.tf_opt_var.set_learning_rate(learning_rate)

        if summ:
            self.variable_summaries()

    def get_optv(self):
        return self.tf_opt_var

    def get_tfv(self):
        if self.fixed:
            return self.tf_opt_var
        else:
            return self.transform.forward_tensor(self.tf_opt_var)

    def variable_summaries(self):
        tf.summary.histogram(self.name,self.tf_opt_var)

    @property
    def shape(self):
        return self.value.shape
