"""
convnet for dqn
"""
import tensorflow as tf

class ConvNetGenerator(object):
    """
    creates, initializes, and returns a convolutional neural network with given params.
    """
    def __init__(self, params, net_input, trainable):
        """
        net_input is a placeholder variable.
        """
        super(ConvNetGenerator, self).__init__()
        self.trainable = trainable     #whether the weights on this network will be trained
        self.input_shape = [None, params.img_height, params.img_width, params.history]  #batch of input images to net
        self.input_dims = self.input_shape[1]*self.input_shape[2]*self.input_shape[3]   #pixels in each image
        self.output_dims = params.output_dims
        self.n_units = params.n_units
        try:
            self.conv_layers = len(params.n_units)   #number of convolutional layers
        except TypeError:
            raise AttributeError("Size of convolutional units undefined")
        if not (self.conv_layers > 0):
            raise ValueError("At least one convolutional layer required!")

        self.filter_size = params.filter_size
        assert(len(self.filter_size) == self.conv_layers)

        self.filter_stride = params.filter_stride
        assert(len(self.filter_stride) == self.conv_layers)

        self.n_hid = params.n_hid
        try:
            self.full_connect_layers = len(params.n_hid)   #number of fully connected layers
        except TypeError:
            raise TypeError("Size of fully connected units undefined")
        if not (self.full_connect_layers > 0):
            raise ValueError("At least one fully connected layer required!")
        #store a dictionary to all weights in network
        self.var_dir = {}
        #scope under which the network was created
        self.scope_name = tf.constant("dummy").name.rsplit('/',1)[0]
        self.logits = self.inference(net_input)
        self.create_weight_cp_ops()

    def create_weights(self, shape):
        """
        creates weights with truncated normal initialization (mean = 0, stddev = 1.0)
        currently created on highest priority available device (cpu or gpu)
        """
        return tf.get_variable('weights', shape,
                        initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.001),
                        trainable=self.trainable)

    def create_bias(self, size):
        """
        creates bias vector of shape [size] filled with 0.1
        """
        return tf.get_variable('bias', [size],
                                initializer=tf.constant_initializer(0.001),
                                trainable=self.trainable)

    def copy_weights(self, other_var_dir, sess):
        for (other_var_name, other_var) in other_var_dir.iteritems():
            var_name = self.scope_name + '/' + other_var_name.split('/', 1)[1]
            other_var_eval = other_var.eval()
            sess.run(self.weight_copy_ops[var_name], feed_dict={self.weight_placeholders[var_name]: other_var_eval})

    def create_weight_cp_ops(self):
        """
        creates an op to overwrite current set of weights
        with weights of another network
        """
        self.weight_placeholders = {}
        for var_name in self.var_dir:
            self.weight_placeholders[var_name] = tf.placeholder(tf.float32)
        self.weight_copy_ops = {}
        for (var_name, var_placeholder) in self.weight_placeholders.iteritems():
            self.weight_copy_ops[var_name] = self.var_dir[var_name].assign(var_placeholder)


    def inference(self, net_input):
        """
        Cnn with self.conv_layers convolutional layers and self.full_connect_layers fully
        connected layers. relu non-linearity after each conv layer. no max-pooling.
        """
        outputs = [net_input]
        #print "INPUT"
        #print outputs[-1].get_shape()
        for conv_layer in range(self.conv_layers):
            with tf.variable_scope('conv' + str(conv_layer)) as scope:
                #create shape of convolutional weight matrix
                if conv_layer == 0: #first layer
                    in_channels = self.input_shape[-1]
                    out_channels = self.n_units[conv_layer]
                else:   #mid layers
                    in_channels = self.n_units[conv_layer - 1]
                    out_channels = self.n_units[conv_layer]
                shape = [self.filter_size[conv_layer], self.filter_size[conv_layer],
                        in_channels, out_channels]
                #print shape
                W = self.create_weights(shape)
                conv = tf.nn.conv2d(outputs[-1], W, [1, self.filter_stride[conv_layer],
                                                        self.filter_stride[conv_layer],1], padding='SAME')
                b = self.create_bias(out_channels)
                self.var_dir[W.name] = W
                self.var_dir[b.name] = b
                bias = tf.nn.bias_add(conv, b)
                conv = tf.nn.relu(bias, name=scope.name)
                outputs.append(conv)
                #print "CONV" + str(conv_layer)
                #print outputs[-1].get_shape()

        last_conv = outputs[-1]
        dim = 1
        for d in last_conv.get_shape()[1:].as_list():
            dim *= d
        reshape = tf.reshape(last_conv, [-1, dim], name='reshape')
        outputs.append(reshape)
        #print "RESHAPED"
        #print outputs[-1].get_shape()
        for connect_layer in range(self.full_connect_layers):
            with tf.variable_scope('hidden' + str(connect_layer)) as scope:
                #find size of weight matrix
                if connect_layer == 0: #first layer
                    in_size = dim
                else:   #mid layers
                    in_size = self.n_hid[connect_layer - 1]
                out_size = self.n_hid[connect_layer]
                shape = [in_size, out_size]
                W = self.create_weights(shape)
                b = self.create_bias(out_size)
                self.var_dir[W.name] = W
                self.var_dir[b.name] = b
                hidden = tf.nn.relu_layer(outputs[-1], W, b, name = scope.name)
                outputs.append(hidden)
                #print "FULLY CONNECTED"
                #print outputs[-1].get_shape()


        #last linear layer connecting to outputs
        with tf.variable_scope('output') as scope:
            in_size = self.n_hid[self.full_connect_layers - 1]
            out_size = self.output_dims
            shape = [in_size, out_size]
            W = self.create_weights(shape)
            b = self.create_bias(out_size)
            self.var_dir[W.name] = W
            self.var_dir[b.name] = b
            hidden = tf.nn.bias_add(tf.matmul(outputs[-1], W), b)
            outputs.append(hidden)
        #print "LAST FULLY CONNECTED"
        #print outputs[-1].get_shape()
        return outputs[-1]  #return linear activations of output
