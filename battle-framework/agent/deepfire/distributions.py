import tensorflow as tf
import numpy as np

class Pd(object):
    """
    A particular probability distribution
    """
    def flatparam(self):
        raise NotImplementedError
    def mode(self):
        raise NotImplementedError
    def neglogp(self, x):
        # Usually it's easier to define the negative logprob
        raise NotImplementedError
    def kl(self, other):
        raise NotImplementedError
    def entropy(self):
        raise NotImplementedError
    def sample(self):
        raise NotImplementedError
    def logp(self, x):
        return - self.neglogp(x)
    def get_shape(self):
        return self.flatparam().shape
    @property
    def shape(self):
        return self.get_shape()
    def __getitem__(self, idx):
        return self.__class__(self.flatparam()[idx])

class CategoricalPd(Pd):
    def __init__(self, logits):
        self.logits = logits
    def flatparam(self):
        return self.logits
    def mode(self):
        return tf.argmax(self.logits, axis=-1)

    @property
    def mean(self):
        return tf.nn.softmax(self.logits)
    def neglogp(self, x):
        # return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=x)
        # Note: we can't use sparse_softmax_cross_entropy_with_logits because
        #       the implementation does not allow second-order derivatives...
        if x.dtype in {tf.uint8, tf.int32, tf.int64}:
            # one-hot encoding
            x_shape_list = x.shape.as_list()
            logits_shape_list = self.logits.get_shape().as_list()[:-1]
            for xs, ls in zip(x_shape_list, logits_shape_list):
                if xs is not None and ls is not None:
                    assert xs == ls, 'shape mismatch: {} in x vs {} in logits'.format(xs, ls)

            # x = tf.one_hot(x, self.logits.get_shape().as_list()[-1])
            x = tf.one_hot(x, self.logits.shape.as_list()[-1])

        else:
            # already encoded
            assert x.shape.as_list() == self.logits.shape.as_list()

        return tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.logits,
            labels=x)
    def kl(self, other):
        pass
        # a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
        # a1 = other.logits - tf.reduce_max(other.logits, axis=-1, keepdims=True)
        # ea0 = tf.exp(a0)
        # ea1 = tf.exp(a1)
        # z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        # z1 = tf.reduce_sum(ea1, axis=-1, keepdims=True)
        # p0 = ea0 / z0
        # return tf.reduce_sum(p0 * (a0 - tf.log(z0) - a1 + tf.log(z1)), axis=-1)
    def entropy(self):
        a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=-1)
    def sample(self):
        u = tf.random_uniform(tf.shape(self.logits), dtype=self.logits.dtype)
        return tf.argmax(self.logits - tf.log(-tf.log(u)), axis=-1)
    @classmethod
    def fromflat(cls, flat):
        return cls(flat)

class DiagGaussianPd(Pd):
    def __init__(self, flat):
        self.flat = flat
        mean, logstd = tf.split(axis=len(flat.shape)-1, num_or_size_splits=2, value=flat)
        self.mean = mean
        self.logstd = logstd
        self.std = tf.exp(logstd)
    def flatparam(self):
        return self.flat
    def mode(self):
        return self.mean
    def neglogp(self, x):
        return 0.5 * tf.reduce_sum(tf.square((x - self.mean) / self.std), axis=-1) \
               + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(x)[-1]) \
               + tf.reduce_sum(self.logstd, axis=-1)
    def kl(self, other):
        pass
        # assert isinstance(other, DiagGaussianPd)
        # return tf.reduce_sum(other.logstd - self.logstd + (tf.square(self.std) + tf.square(self.mean - other.mean)) / (2.0 * tf.square(other.std)) - 0.5, axis=-1)
    def entropy(self):
        return tf.reduce_sum(self.logstd + .5 * np.log(2.0 * np.pi * np.e), axis=-1)
    def sample(self):
        return self.mean + self.std * tf.random_normal(tf.shape(self.mean))
    @classmethod
    def fromflat(cls, flat):
        return cls(flat)


class CategoricalPd_var_mask(Pd):
    '''
        variable length with mask.
    '''
    def __init__(self, logits, mask):
        self.logits = logits
        self.mask = mask
        self.null = tf.constant([[[1e-5]]])

    def flatparam(self):
        return self.logits

    def mode(self):
        return tf.argmax(self.logits, axis=-1)

    @property
    def mean(self):
        return tf.nn.softmax(self.logits)

    def neglogp_all(self):
        a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0) * self.mask
        ea0 = tf.concat([ea0, tf.tile(self.null, [tf.shape(ea0)[0], tf.shape(ea0)[1], 1])], axis=2)

        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return -tf.log(p0 + 1e-9)

    def neglogp(self, x):
        a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0) * self.mask
        ea0 = tf.concat([ea0, tf.tile(self.null, [tf.shape(ea0)[0], 1])], axis=1)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return -tf.reduce_sum(tf.log(p0 + 1e-9) * x, axis=-1)

    #         # return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=x)
    #         # Note: we can't use sparse_softmax_cross_entropy_with_logits because
    #         #       the implementation does not allow second-order derivatives...
    #         if x.dtype in {tf.uint8, tf.int32, tf.int64}:
    #             # one-hot encoding
    #             x_shape_list = x.shape.as_list()
    #             logits_shape_list = self.logits.get_shape().as_list()[:-1]
    #             for xs, ls in zip(x_shape_list, logits_shape_list):
    #                 if xs is not None and ls is not None:
    #                     assert xs == ls, 'shape mismatch: {} in x vs {} in logits'.format(xs, ls)

    #             # x = tf.one_hot(x, self.logits.get_shape().as_list()[-1])
    #             x = tf.one_hot(x, length)

    #         else:
    #             # already encoded
    #             assert x.shape.as_list() == self.logits.shape.as_list()

    #         return softmax_cross_entropy_with_logits_v2(
    #             logits=self.logits,
    #             labels=x)
    def kl(self, other):
        a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
        a1 = other.logits - tf.reduce_max(other.logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        ea1 = tf.exp(a1)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        z1 = tf.reduce_sum(ea1, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (a0 - tf.log(z0) - a1 + tf.log(z1)), axis=-1)

    def entropy(self):
        a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0) * self.mask
        ea0 = tf.concat([ea0, tf.tile(self.null, [tf.shape(ea0)[0], 1])], axis=1)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return -tf.reduce_sum(p0 * tf.log(p0 + 1e-9), axis=-1)

    #         a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
    #         ea0 = tf.exp(a0)
    #         z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
    #         p0 = ea0 / z0
    #         return tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=-1)

    # def sample(self):
    #     u = tf.random_uniform(tf.shape(self.logits), dtype=self.logits.dtype)
    #
    #     exp = tf.exp(self.logits - tf.log(-tf.log(u))) * self.mask
    #
    #     exp = tf.concat([exp, tf.tile(self.null, [tf.shape(exp)[0], 1])], axis=-1)
    #
    #     return tf.argmax(exp, axis=-1)
    def sample(self):
        u = tf.random_uniform(tf.shape(self.logits), dtype=self.logits.dtype)

        exp = tf.exp(self.logits - tf.log(-tf.log(u))) * tf.tile( tf.expand_dims(self.mask,1), [1,tf.shape(self.logits)[1],1] )

        exp = tf.concat([exp, tf.tile(self.null, [tf.shape(exp)[0], tf.shape(exp)[1], 1])], axis=2)

        return tf.argmax(exp, axis=-1)
    @classmethod
    def fromflat(cls, flat):
        return cls(flat)
