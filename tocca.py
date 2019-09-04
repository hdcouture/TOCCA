
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Concatenate
from keras.layers.core import Dense, Dropout
from keras.regularizers import l2, Regularizer
from keras.initializers import Zeros, Identity
from keras.optimizers import Nadam
from keras.layers.normalization import BatchNormalization
from keras.engine import InputSpec, Layer
from keras import initializers
if 'theano' in dir(K):
    BACKEND = 'theano'
    from theano.tensor import diagonal as diag_part
    from theano.tensor.nlinalg import diag, eigh
    from theano.tensor import inv as reciprocal
    from theano.tensor import nonzero
    from theano.tensor import identity_like as eye_like
else:
    BACKEND = 'tensorflow'
    from tensorflow.linalg import tensor_diag_part as diag_part
    from tensorflow.linalg import tensor_diag as diag
    from tensorflow.linalg import eigh
    from tensorflow.math import reciprocal
    from tensorflow import boolean_mask as nonzero
    import tensorflow as tf
    def eye_like(C):
        return K.eye(K.shape(C)[0])

eps = 1e-12
class ZCA(Layer):
    """ZCA whitening layer."""

    def __init__(self, momentum, r=1e-3, **kwargs):
        super(ZCA,self).__init__(**kwargs)
        self.momentum = K.cast_to_floatx(momentum)
        if r == True:
            r = 1e-3
        self.r = K.cast_to_floatx(r)
        self.initialized = False
        
    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        dim = input_shape[-1]
        shape = (dim,)
        shape = tuple([1]*(len(input_shape)-1)+[input_shape[-1]])

        self.C = self.add_weight(shape=(input_dim,input_dim),
                                 initializer=Zeros(),
                                 name='C',
                                 trainable=False)
        self.U = self.add_weight(shape=(input_dim,input_dim),
                                 initializer=Identity(),
                                 name='U',
                                 trainable=False)
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, X, training=None):

        X0 = K.dot( X, self.U )
        if training in {0,False}:
            return X0

        nd = K.shape(X)[1]
        n = K.shape(X)[0]
        C = K.dot( K.transpose(X), X ) / K.cast(n-1,'float32')
        self.C = self.momentum * self.C + (1-self.momentum) * C

        C = C + self.r * eye_like(C)

        [D,V] = eigh(C)

        # Added to increase stability
        if BACKEND == 'theano':
            posInd = K.greater(D, eps).nonzero()[0]
            D = D[posInd]
            V = V[:, posInd]
        else:
            posBool = K.greater(D,eps)
            D = tf.boolean_mask( D, posBool )
            V = tf.boolean_mask( V, posBool, axis=1 )

        U = K.dot( K.dot( V, diag( reciprocal( K.sqrt( D ) ) ) ), K.transpose(V) )
        U = K.transpose(U)

        self.add_update([(self.U,U)],X)

        X_updated = K.dot( X, U )

        return K.in_train_phase(X_updated,X0,training=training)
    
class SDL(Regularizer):
    """Stochastic decorrelation loss.  From Chang, 2018 paper"""

    def __init__(self, d, momentum, C, l1=0., l2=0.):
        self.d = d
        self.momentum = momentum
        self.C = C
        self.l1 = K.cast_to_floatx(l1)
        self.l2 = K.cast_to_floatx(l2)
        self.denom = 0
        self.initialized = False

    def __call__(self, X):
        Ci = K.dot( K.transpose(X), X ) / (K.cast(K.shape(X)[0],'float32')-1+1e-6)
        if not self.initialized:
            C = 0.0 * self.C + Ci
            self.initialized = True
        else:
            C = self.momentum * self.C + (1-self.momentum) * Ci
        reg = self.l1 * ( K.sum(K.sum(K.abs(C))) - K.sum(K.abs(diag_part(C))) ) + self.l2 * ( K.sum(K.sum( C**2 )) - K.sum(diag_part(C)**2) )
        self.C = C
        return reg

    def get_config(self):
        return {'d': int(self.d),
                'momentum': float(self.momentum),
                'l1': float(self.l1),
                'l2': float(self.l2)}

class StochasticDecorrelation(Layer):
    """Layer for Stochastic decorrelation loss.  From Chang, 2018 paper"""

    def __init__(self, d, momentum, l1=0., l2=0., **kwargs):
        super(StochasticDecorrelation, self).__init__(**kwargs)
        self.supports_masking = True
        self.d = d
        self.momentum = momentum
        self.l1 = l1
        self.l2 = l2
        self.C = self.add_weight(shape=(d,d),
                                 initializer=Zeros(),
                                 name='C',
                                 trainable=False,
                                 constraint=None)
        self.activity_regularizer = SDL(d=d, momentum=momentum, C=self.C, l1=l1, l2=l2)

    def get_config(self):
        config = {'d': self.d,
                  'momentum': self.momentum,
                  'l1': self.l1,
                  'l2': self.l2}
        base_config = super(StochasticDecorrelation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class BatchNorm(Layer):
    """Batch normalization without extra translation and scaling weights."""

    def __init__(self,
                 axis=-1,
                 momentum=0.99,
                 epsilon=1e-6,
                 moving_mean_initializer='zeros',
                 moving_variance_initializer='ones',
                 **kwargs):
        super(BatchNorm, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.moving_mean_initializer = initializers.get(moving_mean_initializer)
        self.moving_variance_initializer = initializers.get(moving_variance_initializer)
        self.initialized = False

    def build(self, input_shape):
        dim = input_shape[self.axis]
        self.input_spec = InputSpec(ndim=len(input_shape),
                                    axes={self.axis: dim})
        shape = (dim,)
        shape = tuple([1]*(len(input_shape)-1)+[input_shape[-1]])

        self.moving_mean = self.add_weight(
            shape=(dim,),
            name='moving_mean',
            initializer=self.moving_mean_initializer,
            trainable=False)
        self.moving_variance = self.add_weight(
            shape=(dim,),
            name='moving_variance',
            initializer=self.moving_variance_initializer,
            trainable=False)
        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)

        X0 = (inputs-self.moving_mean)/K.sqrt(self.moving_variance+self.epsilon)
        if training in {0,False}:
            return X0

        mean = K.mean( inputs, axis=0 )
        variance = K.var( inputs, axis=0 )

        mean = self.momentum * self.moving_mean + (1-self.momentum) * mean
        variance = self.momentum * self.moving_variance + (1-self.momentum) * variance

        self.add_update( [(self.moving_mean,mean),(self.moving_variance,variance)], inputs )

        X_updated = (inputs-mean)/K.sqrt(variance+self.epsilon)
        
        return K.in_train_phase(X_updated,X0,training=training)

    def get_config(self):
        config = {
            'axis': self.axis,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
        }
        base_config = super(BatchNorm, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

_EPSILON = 10e-8
def categorical_crossentropy_missing(target, output):
    """Categorical crossentropy loss, but ignore any samples with no label"""
    # scale preds so that the class probas of each sample sum to 1
    output /= (K.sum(output, axis=1, keepdims=True)+_EPSILON)
    # avoid numerical instability with _EPSILON clipping
    output = K.clip(output, _EPSILON, 1.0 - _EPSILON)
    # identify samples with label (should have 1 for one of the classes)
    select = K.cast( K.greater(K.max(target,axis=1),0.5), 'float32' )
    ce = -K.sum(target * K.log(output), axis=1)
    # only sum across samples with label
    return K.sum( ce * select ) / (K.sum(select)+_EPSILON)

def l2dist( X, ncomp=None ):
    """Squared Frobenius matrix norm of distance between two modalities (left and right half of X)."""

    shape = K.shape(X)
    d = shape[1]//2
    m = shape[0]
    if ncomp is None:
        H1 = X[:,:d]
        H2 = X[:,d:]
    else:
        H1 = X[:,:ncomp]
        H2 = X[:,d:d+ncomp]
    H1bar = H1
    H2bar = H2
    
    diff = H1bar - H2bar
    return K.sum( K.sum( diff**2 ) ) / ( K.cast( K.shape(diff)[0], 'float32') )

def create_model( model_type, nclasses, input_dims, layers, layer_size, shared_size, lr=1e-4, l2dist_weight=1.0, momentum=0.99, l2_weight=0, sd_weight=0, zca_r=1e-4, dropout=None ):
    """Create deep TOCCA model."""

    if BACKEND == 'tensorflow':
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.1
        K.tensorflow_backend.set_session(tf.Session(config=config))
    
    if type(layer_size) is not list:
        layer_size = [ [ layer_size for l in range(layers) ] for m in range(len(input_dims)) ]
    elif type(layer_size[0]) is not list:
        layer_size = [ [ ls for ls in range(layers) ] for ls in layer_size ]

    # create individual layers
    inputs = []
    xindiv = []
    for m,(dim,cur_layer_size) in enumerate(zip(input_dims,layer_size)):
        x = Input(shape=(dim,))
        inputs.append( x )

        # dense layers
        for layer,ls in enumerate(cur_layer_size):
            x = Dense(ls, activation='relu', kernel_regularizer=l2(l2_weight), name='dense_'+str(m)+'_'+str(layer))(x)
            x = BatchNormalization(momentum=momentum)(x)
            if dropout is not None:
                x = Dropout(dropout)(x)

        # shared layer
        kernel_reg = l2(l2_weight)
        x = Dense(shared_size, use_bias=False, kernel_regularizer=kernel_reg, name='dense_'+str(m))(x)
        x = BatchNorm(momentum=momentum)(x)
        
        # apply whitening or soft decorrelation
        if model_type == 'w':
            x = ZCA(momentum=momentum, r=zca_r, name='zca_'+str(m))(x)
        elif model_type == 'sd':
            x = StochasticDecorrelation( shared_size, momentum, l1=sd_weight )(x)

        xindiv.append( x )

    xmerge = Concatenate()(xindiv)

    # softmax output, shared across modalities
    outputs = []
    dense = Dense(nclasses, activation='softmax', kernel_regularizer=l2(l2_weight), name='softmax')
    for m,x in enumerate(xindiv):
        softmax = dense(x)
        outputs.append( softmax )

    losses = [categorical_crossentropy_missing]*len(outputs)
    metrics = ['accuracy']

    outputs.append( xmerge )
    l2dist_loss = lambda y_true, y_pred: l2dist(y_pred,shared_size) * l2dist_weight
    losses.append( l2dist_loss )

    model = Model(inputs=inputs, outputs=outputs)

    print(model.summary())

    model.compile( optimizer=Nadam(lr=lr), loss=losses, metrics=metrics )

    return model
