# -*- coding: utf-8 -*-
"""Show, Ask, Attend, and Answer VQA model for Keras.

# Reference:
    
- [Show, Ask, Attend, and Answer: A Strong Baseline For Visual Question Answering](https://arxiv.org/abs/1704.03162)


"""

import keras.backend as K
from keras.layers import Input
from keras.models import Model
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import AveragePooling2D
from keras.layers.merge import concatenate, multiply
from keras.layers.core import Dense, Dropout, RepeatVector, Reshape, Activation, Lambda, Flatten

def show_ask_attend_answer(vocab_size, num_glimpses=2, n=14):
    # Define network inputs where n is the feature rows and columns. In the
    # paper they use ResNet 152 res5c features with size (14x14x2048)
    image_input = Input(shape=(n,n,2048))
    question_input = Input(shape=(15,))
    
    # Learn word embeddings in relation to total vocabulary
    question_embedding = Embedding(vocab_size, 300, input_length=15)(question_input)
    question_embedding = Activation('tanh')(question_embedding)
    question_embedding = Dropout(0.5)(question_embedding)
    
    # LSTM to seuqentially embed word vectors into a single question vector
    question_lstm = LSTM(1024)(question_embedding)
    
    # Repeating and tiling question vector to match image input for concatenation
    question_tile = RepeatVector(n*n)(question_lstm)
    question_tile = Reshape((n,n,1024))(question_tile)
    
    # Concatenation of question vector and image features
    concatenated_features1 = concatenate([image_input, question_tile])
    concatenated_features1 = Dropout(0.5)(concatenated_features1)
    
    # Stacked attention network
    attention_conv1 = Conv2D(512, (1,1))(concatenated_features1)
    attention_relu = Activation('relu')(attention_conv1)
    attention_relu = Dropout(0.5)(attention_relu)
    
    attention_conv2 = Conv2D(num_glimpses, (1,1))(attention_relu)
    attention_maps = Activation('softmax')(attention_conv2)
    
    # Weighted average of image features using attention maps
    image_attention = glimpse(attention_maps, image_input, num_glimpses, n)
    
    # Concatenation of question vector and attended image features
    concatenated_features2 = concatenate([image_attention, question_lstm])
    concatenated_features2 = Dropout(0.5)(concatenated_features2)
    
    # First fully connected layer with relu and dropout
    fc1 = Dense(1024)(concatenated_features2)
    fc1_relu = Activation('relu')(fc1)
    fc1_relu = Dropout(0.5)(fc1_relu)
    
    # Final fully connected layer with softmax to output answer probabilities
    fc2 = Dense(3000)(fc1_relu)
    fc2_softmax = Activation('softmax')(fc2)
    
    # Instantiate the model
    vqa_model = Model(inputs=[image_input, question_input], outputs=fc2_softmax)
    
    return vqa_model

def glimpse(attention_maps, image_features, num_glimpses=2, n=14):
    glimpse_list = []
    for i in range(num_glimpses):
        glimpse_map = Lambda(lambda x: x[:,:,:,i])(attention_maps)                # Select the i'th attention map
        glimpse_map = Reshape((n,n,1))(glimpse_map)                               # Reshape to add channel dimension for K.tile() to work. (14,14) --> (14,14,1)
        glimpse_tile = Lambda(tile)(glimpse_map)                                  # Repeat the attention over the channel dimension. (14,14,1) --> (14,14,2048)
        weighted_features = multiply([image_features, glimpse_tile])              # Element wise multiplication to weight image features
        weighted_average = AveragePooling2D(pool_size=(n,n))(weighted_features) # Average pool each channel. (14,14,2048) --> (1,1,2048)
        weighted_average = Flatten()(weighted_average)
        glimpse_list.append(weighted_average)
        
    return concatenate(glimpse_list)

def tile(x):
    return K.tile(x, [1,1,1,2048])

if __name__ == '__main__':
    vqa_model = show_ask_attend_answer(vocab_size=15000) # change vocab size to the real value. 15000 is just an estimate from what I used in past VQA models
    vqa_model.summary()