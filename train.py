from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from keras.models import load_model
import keras
from models import ModelLoader
from data import CNN3DDataLoader
import time

def save_history(history,  name):
    print (history)
    loss=history.history['loss']
    acc=history.history['acc']
    val_loss=history.history['val_loss']
    val_acc=history.history['val_acc']
    nb_epoch=len(acc)

    with open(os.path.join("/Users/benja/code/jester/result", 'result_{}.txt'.format(name)), 'w') as fp:
        fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                i, loss[i], acc[i], val_loss[i], val_acc[i]))


labels_want = ['Swiping Left',
          'Swiping Right',
          'Swiping Down',
          'Swiping Up']

def train(model_name):
    # Data parameters
    #data_dir = r'C:\Users\Lorenz\Documents\Coding\data\jester_hand_gestures'
    data_dir = r'/Users/benja/code/jester/data'
    seq_length = 40
    n_videos = {'train': 1000, 'validation': 100}
    image_size=(50, 88)

    # Training parameters
    n_epochs = 50
    batch_size = 8
    steps_per_epoch = n_videos['train'] // batch_size

    # Load data generators
    data = CNN3DDataLoader(data_dir, seq_length=seq_length, n_videos=n_videos, labels = labels_want)
    train_gen = data.sequence_generator('train', batch_size, image_size)
    validation_gen = data.sequence_generator('validation', batch_size, image_size)

    #optimizer = keras.optimizers.SGD(lr=0.1, momentum=0.9, decay= 1e-6, nesterov=True)
    #Load model
    optimizer = keras.optimizers.Adadelta()
    ml = ModelLoader(data.n_labels, data.seq_length, model_name, image_size=image_size, optimizer = optimizer)
    model = ml.model

    #Define callbacks
    checkpointer = ModelCheckpoint(
        filepath='model_name' + '-{epoch:03d}-{loss:.3f}.hdf5',
        verbose=1,
        save_best_only=True)
    tb = TensorBoard(log_dir='./models/logs')
    early_stopper = EarlyStopping(patience=2)
    csv_logger = CSVLogger('./models/logs/' + model_name + '-' + 'training-' + \
        str(time.time()) + '.log')

    callbacks = [tb, early_stopper, csv_logger, checkpointer]

    # Training
    print('Starting training')

    history = model.fit_generator(
        generator=train_gen,
        steps_per_epoch=20,
        #sample_per_epoch= 200,
        epochs=n_epochs,
        verbose=1,
        callbacks=callbacks,
        validation_data=validation_gen,
        validation_steps=10,
    )


    model.save('./my_model.h5')

    save_history(history,"c3d")



if __name__=='__main__':
    #train("c3d")
    train("small_c3d")
