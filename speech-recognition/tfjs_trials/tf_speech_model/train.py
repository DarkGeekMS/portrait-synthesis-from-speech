import sys
import os
import keras
from keras import backend as K
from keras.models import Model
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint   
from keras.layers import (BatchNormalization, Input,Lambda, TimeDistributed, Activation, Dense, Bidirectional, GRU, LSTM, Dropout, MaxPooling1D)
import tensorflowjs as tfjs
from DataGenerator import AudioGenerator


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def add_ctc_loss(input_to_softmax):
    the_labels = Input(name='the_labels', shape=(None,), dtype='float32')
    input_lengths = Input(name='input_length', shape=(1,), dtype='int64')
    label_lengths = Input(name='label_length', shape=(1,), dtype='int64')
    output_lengths = Lambda(input_to_softmax.output_length)(input_lengths)
    # CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
        [input_to_softmax.output, the_labels, output_lengths, label_lengths])
    model = Model(
        inputs=[input_to_softmax.input, the_labels, input_lengths, label_lengths], 
        outputs=loss_out)
    return model

def model1(input_dim, units, activation, output_dim=29):
    """ 
        Recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    gru_layer = GRU(units, activation=activation, return_sequences=True, implementation=2, name='rnn')(input_data)
    # Add batch normalization 
    bn_gru_layer = BatchNormalization(name = 'bn_gru_layer')(gru_layer)
    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_gru_layer)
    # Add softmax activation layer
    y_pred = Activation('softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model



def model2(input_dim, units, output_dim=29):
    """ 
        bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    bidir_rnn = Bidirectional(GRU(units, activation='relu',return_sequences=True, implementation=2, name='bidir_rnn'))(input_data)
    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model



def train_model(input_model,                            # the specified model
                save_model_path,                        # the weights path
                train_json='train_corpus.json',         # training data
                valid_json='valid_corpus.json',         # validation data
                minibatch_size=20,                      # batch size
                spectrogram=True,                       #default preprocessing
                mfcc_dim=161,
                # the optimizer
                optimizer=SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5),
                epochs=50,                              # number of epochs
                verbose=1,
                sort_by_duration=False,
                max_duration=10.0):
    
    # create a class instance for obtaining batches of data
    audio_gen = AudioGenerator(minibatch_size=minibatch_size, spectrogram=spectrogram, mfcc_dim=mfcc_dim, max_duration=max_duration, sort_by_duration=sort_by_duration)
    
    # add the training data to the generator
    audio_gen.load_train_data(train_json)
    audio_gen.load_validation_data(valid_json)

    # calculate steps_per_epoch
    num_train_examples=len(audio_gen.train_audio_paths)
    steps_per_epoch = num_train_examples//minibatch_size

    # calculate validation_steps
    num_valid_samples = len(audio_gen.valid_audio_paths) 
    validation_steps = num_valid_samples//minibatch_size
    
    # add CTC loss to the specified model
    model = add_ctc_loss(input_model)

    # CTC loss is implemented elsewhere, so use a dummy lambda function for the loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)

    # make results/ directory, if necessary
    if not os.path.exists('results'):
        os.makedirs('results')

    # add checkpointer
    checkpointer = ModelCheckpoint(filepath='results/'+save_model_path, verbose=0)
    tfjs.converters.save_keras_model(model, "weights/model.json")

    # train the model
    hist = model.fit_generator(generator=audio_gen.next_train(), steps_per_epoch=steps_per_epoch,
        epochs=epochs, validation_data=audio_gen.next_valid(), validation_steps=validation_steps,
        callbacks=[checkpointer], verbose=verbose)


def main(model = "m1", feature = "spec" ):

    use_spec = False
    if feature == "spec" :
        f = 161
        use_spec = True
    else:
        f = 13
        use_spec = False
    if model == "m1":
        model_1 = model1(input_dim=f, units=200, activation='relu')
        train_model(input_model=model_1, save_model_path='model_1.h5', spectrogram=use_spec)
    else:
        model_2 = model2(input_dim=f, units=200)
        train_model(input_model=model_2, save_model_path='model_4.h5', spectrogram=use_spec)


if __name__ == '__main__':
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        raise Exception("Error number of parameters")

    
