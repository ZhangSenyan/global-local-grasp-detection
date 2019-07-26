


import utils


from keras.models import Sequential
from keras.layers import Dense,Activation,Convolution2D,MaxPooling2D,Flatten
from keras.optimizers import RMSprop,Adam




def main():
    data_path_base='/home/zhsy/work/DataSet/grasp2/'
    (X_train, Y_train), (X_test, Y_test)=utils.load_patches(data_path_base)

    print(X_train.shape)

    # build model
    model = Sequential()

    model.add(Convolution2D(nb_filter=32,
                            nb_row=5,
                            nb_col=5,
                            border_mode='same',
                            input_shape=(36,72, 1)
                            ))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        border_mode='same'
    ))
    model.add(Convolution2D(nb_filter=64,
                            nb_row=5,
                            nb_col=5,
                            border_mode='same'
                            ))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        border_mode='same'
    ))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    # define optimizer

    rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    adam = Adam(lr=1e-4)

    # compile
    model.compile(
        optimizer=adam,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    msg=[]

    for i in range(5):
        print("Training-----" )
        model.fit(X_train, Y_train, batch_size=32, nb_epoch=10)

        print('Testing-----')
        loss, accuracy = model.evaluate(X_test, Y_test)
        print("nb_epoch=%d" % (10 * i))
        print('test_loss=', loss)
        print('accuracy=', accuracy)
        msg.append(accuracy)

    print(msg)

    model.save('./model/local_cnn.h5')


if __name__ == '__main__':
    main()