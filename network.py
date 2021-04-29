from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed, Activation, Bidirectional, Masking


def building_ner(num_lstm_layer, num_hidden_node, dropout, time_step, vector_length, output_lenght):
    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=(time_step, vector_length)))
    '''
        X.shape=(samples, timesteps, features) #(data size, len(sequence), #dim)
        Masking: tell sequence-processing layers that certain timesteps in an input are missing, 
                    and thus should be skipped when processing the data.
        '''
    for i in range(num_lstm_layer-1):
        model.add(Bidirectional(LSTM(units=num_hidden_node, return_sequences=True, dropout=dropout,
                                     recurrent_dropout=dropout)))
        '''
        return_sequences: used for stacked LSTM (return the last output if False)
        dropout vs recurrent_dropout:
            https://stackoverflow.com/questions/44924690/keras-the-difference-between-lstm-dropout-and-lstm-recurrent-dropout
        '''
    model.add(Bidirectional(LSTM(units=num_hidden_node, return_sequences=True, dropout=dropout,
                                 recurrent_dropout=dropout), merge_mode='concat'))
    model.add(TimeDistributed(Dense(output_lenght)))
    '''
    TimeDistributed: apply Dense for each time step
    output_length: #labels
    '''
    model.add(Activation('softmax'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model
