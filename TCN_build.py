from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalMaxPooling1D, GlobalAveragePooling1D, Concatenate, Dense, Dropout
from tcn import TCN 

def create_model():
    inputs = Input(shape=(15, 1))
    tcn = TCN(
        nb_filters=128,
        nb_stacks=2,
        kernel_size=5,
        dilations=[4, 8, 16, 32, 64],
        dropout_rate=0.05,
        use_layer_norm=True,
        return_sequences=True
    )(inputs)
    
    max_pooling = GlobalMaxPooling1D()(tcn)
    avg_pooling = GlobalAveragePooling1D()(tcn)
    pooling = Concatenate()([max_pooling, avg_pooling])
    dense1 = Dense(64, activation='relu')(pooling)
    dropout1 = Dropout(0.2)(dense1)
    outputs = Dense(2, activation='softmax')(dropout1)

    model = Model(inputs=inputs, outputs=outputs)
    return model
