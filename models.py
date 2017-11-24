from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv3D, MaxPooling3D


class ModelLoader():

    def __init__(self, n_labels, seq_length, model_name,
                 saved_weights=None, optimizer=None, image_size=(100, 176)):

        self.n_labels = n_labels
        self.load_model = load_model
        self.saved_weights = saved_weights
        self.model_name = model_name

        # Loads the specified model
        if self.model_name == 'small_c3d':
            print('Loading Small C3D model')
            self.input_shape = ((seq_length,) + image_size + (3,) )
            self.model = self.small_c3d()

        elif self.model_name == 'smaller_c3d':
            print('Loading Smaller C3D model')
            self.input_shape = ((seq_length,) + image_size + (3,) )
            self.model = self.smaller_c3d()

        elif self.model_name == "c3d":
            print('Loading  C3D model')
            self.input_shape = ((seq_length,) + image_size + (3,) )
            self.model = self.c3d()

        elif self.model_name == "big_c3d":
            print('loading Big C3D model')
            self.input_shape = ((seq_length,) + image_size + (3,) )
            self.model = self.big_c3d()
        elif self.model_name == "small_c3dv2":
            print('loading Smaller C3D model v2')
            self.input_shape = ((seq_length,) + image_size + (3,) )
            self.model = self.small_c3dv2()

        else:
            raise Exception('No model with name {} found!'.format(model_name))
        # Define metrics
        metrics = ['accuracy', 'top_k_categorical_accuracy']

        # If no optimizer is given, use Adam as default
        if not optimizer:
            optimizer = Adam()

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=optimizer,
                           metrics=metrics)
        print(self.model.summary())

    def small_c3dv2(self):
        """See: 'https://arxiv.org/pdf/1412.0767.pdf' """
        # Tunable parameters
        kernel_size = (3, 3, 3)
        strides = (1, 1, 1)
        extra_conv_blocks = 1

        model = Sequential()

        # Conv Block 1
        model.add(Conv3D(32, (7,7,5), strides=strides, activation='relu',
                         padding='same', input_shape=self.input_shape))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))

        # Conv Block 2
        model.add(Conv3D(64, (5,5,3), strides=strides, activation='relu',
                         padding='same'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

        # Conv Block 3
        model.add(Conv3D(128, (5,5,3), strides=strides, activation='relu',
                         padding='same'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))
        #model.add(Dropout(0.25))

        # Conv Block 4
        model.add(Conv3D(128, (3,5,3), strides=strides, activation='relu',
                         padding='same'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))
        model.add(Dropout(0.25))

        # Dense Block
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.n_labels, activation='softmax'))

        return model

    def small_c3d(self):
        """See: 'https://arxiv.org/pdf/1412.0767.pdf' """
        # Tunable parameters
        kernel_size = (3, 3, 3)
        strides = (1, 1, 1)
        extra_conv_blocks = 1

        model = Sequential()

        # Conv Block 1
        model.add(Conv3D(32, kernel_size, strides=strides, activation='relu',
                         padding='same', input_shape=self.input_shape))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))

        # Conv Block 2
        model.add(Conv3D(64, kernel_size, strides=strides, activation='relu',
                         padding='same'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

        # Conv Block 3
        model.add(Conv3D(128, kernel_size, strides=strides, activation='relu',
                         padding='same'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

        # Conv Block 4
        model.add(Conv3D(128, kernel_size, strides=strides, activation='relu',
                         padding='same'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

        # Dense Block
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(self.n_labels, activation='softmax'))

        return model

    def big_c3d(self):
        kernel_size = (3, 3, 3)
        strides = (1, 1, 1)

        model = Sequential()

        # Conv Block 1
        model.add(Conv3D(64, kernel_size, strides=strides, activation='relu',
                         padding='same', input_shape=self.input_shape))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))

        # Conv Block 2
        model.add(Conv3D(128, kernel_size, strides=strides, activation='relu',
                         padding='same'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

        # Conv Block 3
        model.add(Conv3D(256, kernel_size, strides=strides, activation='relu',
                         padding='same'))
        #model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))
        model.add(Conv3D(256, kernel_size, strides=strides, activation='relu',
                         padding='same'))

        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))
        # Conv Block 4
        model.add(Conv3D(512, kernel_size, strides=strides, activation='relu',
                         padding='same'))

        model.add(Conv3D(512, kernel_size, strides=strides, activation='relu',
                         padding='same'))# Dense Block
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

        #Conv Block 5
        model.add(Conv3D(512, kernel_size, strides=strides, activation='relu',
                         padding='same'))
        model.add(Conv3D(512, kernel_size, strides=strides, activation='relu',
                         padding='same'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))
        
        model.add(Flatten())
        model.add(Dense(2048, activation='relu'))
        model.add(Dense(2048, activation='relu'))
        model.add(Dense(self.n_labels, activation='softmax'))
        return model

    def c3d(self):
        """See: 'https://arxiv.org/pdf/1412.0767.pdf' """
        # Tunable parameters
        strides = (1, 1, 1)
        model = Sequential()
        model.add(Conv3D(32, kernel_size=(3, 3, 3), strides=strides,input_shape=(
           self.input_shape), border_mode='same', activation='relu'))
        #model.add(Activation('relu'))

        model.add(Conv3D(32, kernel_size=(3, 3, 3), strides=strides,padding='same', activation='softmax'))
        #model.add(Activation('softmax'))

        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same'))
        model.add(Dropout(0.25))

        model.add(Conv3D(64, kernel_size=(3, 3, 3),strides=strides, padding='same', activation='relu'))
        #model.add(Activation('relu'))
        model.add(Conv3D(64, kernel_size=(3, 3, 3),strides=strides, padding='same', activation='softmax'))
        #model.add(Activation('softmax'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same'))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512, activation='sigmoid'))
        model.add(Dropout(0.5))
        model.add(Dense(self.n_labels, activation='softmax'))


        return model

    def smaller_c3d(self):
        model = Sequential()
        model.add(Conv3D(
            32, (7,7,7), activation='relu', input_shape=self.input_shape
        ))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
        model.add(Conv3D(64, (3,3,3), activation='relu'))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
        model.add(Conv3D(128, (2,2,2), activation='relu'))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
        model.add(Flatten())
        model.add(Dense(256))
        model.add(Dropout(0.2))
        model.add(Dense(256))
        model.add(Dropout(0.2))
        model.add(Dense(self.n_labels, activation='softmax'))

        return model
