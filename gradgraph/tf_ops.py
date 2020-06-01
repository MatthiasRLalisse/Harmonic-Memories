import tensorflow as tf
import string
import uuid
no_zeros = 10e-10

def cconv(x, y):
	x_fft_ = tf.fft(tf.complex(x,0.0))
	#e2_fft_ = tf.fft(tf.complex(tf.nn.l2_normalize(self.e2, axis=2),0.0))
	y_fft_ = tf.fft(tf.complex(y,0.0))
	x_fft = x_fft_ #+ tf.complex(tf.to_float(tf.equal(x_fft_, 0.)),0.)*no_zeros
	y_fft = y_fft_ #+ tf.complex(tf.to_float(tf.equal(y_fft_, 0.)),0.)*no_zeros
	return tf.cast(tf.real(tf.ifft(tf.multiply(tf.conj(x_fft),\
                                             y_fft))),dtype=tf.float32)

def ccorr(x, y):
	x_fft_ = tf.fft(tf.complex(x,0.0))
	#e2_fft_ = tf.fft(tf.complex(tf.nn.l2_normalize(self.e2, axis=2),0.0))
	y_fft_ = tf.fft(tf.complex(y,0.0))
	x_fft = x_fft_ #+ tf.complex(tf.to_float(tf.equal(x_fft_, 0.)),0.)*no_zeros
	y_fft = y_fft_ #+ tf.complex(tf.to_float(tf.equal(y_fft_, 0.)),0.)*no_zeros
	return tf.cast(tf.real(tf.ifft(tf.multiply(x_fft,\
                                             y_fft))),dtype=tf.float32)


def bilinearForm(x, M, y):
	xM = tf.tensordot(x, M, [-1, 0])
	xMy = tf.reduce_sum(tf.multiply(xM, y), axis=-1)
	return xMy

def quadraticForm(x, M):
	return bilinearForm(x, M, x)
	

@tf.RegisterGradient("HeavisideGrad")
def _heaviside_grad(unused_op: tf.Operation, grad: tf.Tensor):
    return tf.maximum(0.0, 1.0 - tf.abs(unused_op.inputs[0])) * grad


def heaviside(x: tf.Tensor, g: tf.Graph = tf.get_default_graph()):
    custom_grads = {
        "Identity": "HeavisideGrad"
    }
    with g.gradient_override_map(custom_grads):
        i = tf.identity(x, name="identity_" + str(uuid.uuid1()))
        ge = tf.greater_equal(x, 0, name="ge_" + str(uuid.uuid1()))
        # tf.stop_gradient is needed to exclude tf.to_float from derivative
        step_func = i + tf.stop_gradient(tf.to_float(ge) - i)
        return step_func


