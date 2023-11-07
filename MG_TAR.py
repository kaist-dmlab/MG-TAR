import tensorflow as tf

from tensorflow.keras import layers
from tensorflow import keras

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Activation, MultiHeadAttention, Dropout, RepeatVector, TimeDistributed
from tensorflow.keras.layers import Concatenate, Lambda, Reshape, GRU, BatchNormalization, Dot, Add, Bidirectional

from spektral.layers import GCNConv, GlobalAvgPool

es = tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True)
lr = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.001, decay_steps=10000, decay_rate=0.9)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
def InterviewAttention(V, H):
    V_attn = Dense(V.shape[1], activation='relu')(V)
    V_attn = Dense(V_attn.shape[1], activation='sigmoid')(V_attn)
    return Dot(axes=-1)([H, V_attn])

def TemporalAttention(h_units, H, length):
    H = TransformerBlock(H.shape[-1], length, 2048)(H) # embed_dim, num_heads, ff_dim
    H = Reshape(target_shape=[length, -1])(H)
    return GRU(h_units)(H)

def MG_TAR(x_train, y_train, x_val, y_val, configs, length=12, n_steps=6):
    tf.keras.backend.clear_session()
    
    _, _, _, _, _, node_features = x_train
    _, _, n_districts, n_features = node_features.shape
    gru_h, gcn_f, fc_h, n_layers, bn, d = configs

    A_S = Input(shape=[n_districts, n_districts]) # spatial closeness
    A_P = Input(shape=[n_districts, n_districts]) # functional similarity (POI)
    A_R = Input(shape=[n_districts, n_districts]) # road similarity
    A_D = Input(shape=[n_districts, n_districts]) # demographic similarity
    A_T = Input(shape=[length, n_districts, n_districts]) # traffic patterns

    F = Input(shape=[length, n_districts, n_features]) # node features
    
    H = [] # H_1 to H_T

    for t in range(length):
        # slice for each time step t
        Ft = Lambda(lambda f: f[:,t,:,:])(F)
        A_Tt = Lambda(lambda a: a[:,t,:,:])(A_T)

        X_S, X_P, X_R, X_D, X_Tt = Ft, Ft, Ft, Ft, Ft # input H_t0 time t layer 0
        H_S, H_P, H_R, H_D, H_Tt = X_S, X_P, X_R, X_D, X_Tt

        for i in range(n_layers): # using aggregation for each layer as in ST-MGCN ?
            H_S = GCNConv(gcn_f)([H_S, A_S]) # GCN for Adjacency Matrix
            if (i + 1) % 2 == 0:
                H_S = BatchNormalization()(H_S) if bn else H_S
            H_S = Activation('relu')(H_S)

            H_P = GCNConv(gcn_f)([H_P, A_P]) # GCN for POI Graph
            if (i + 1) % 2 == 0:
                H_P = BatchNormalization()(H_P) if bn else H_P
            H_P = Activation('relu')(H_P)

            H_D = GCNConv(gcn_f)([H_D, A_D])  # GCN for Demographic Graph
            if (i + 1) % 2 == 0:
                H_D = BatchNormalization()(H_D) if bn else H_D
            H_D = Activation('relu')(H_D)

            H_R = GCNConv(gcn_f)([H_R, A_R]) # GCN for Road Graph
            if (i + 1) % 2 == 0:
                H_R = BatchNormalization()(H_R) if bn else H_R
            H_R = Activation('relu')(H_R)   

            H_Tt = GCNConv(gcn_f)([H_Tt, A_Tt]) # GCN for Traffic Patterns
            if (i + 1) % 2 == 0:
                H_Tt = BatchNormalization()(H_Tt) if bn else H_Tt
            H_Tt = Activation('relu')(H_Tt) 

            
        H_S = GCNConv(1, activation='relu')([H_S, A_S]) 
        H_P = GCNConv(1, activation='relu')([H_P, A_P])
        H_D = GCNConv(1, activation='relu')([H_D, A_D])
        H_R = GCNConv(1, activation='relu')([H_R, A_R])
        H_Tt = GCNConv(1, activation='relu')([H_Tt, A_Tt])
        
        # summarize each channel (i.e., view) into a scalar 
        z = Concatenate()([GlobalAvgPool()(H_S), GlobalAvgPool()(H_P), GlobalAvgPool()(H_D), GlobalAvgPool()(H_R), GlobalAvgPool()(H_Tt)]) # concatenate it into vector z
        Ht = Concatenate()([H_S, H_P, H_D, H_R, H_Tt]) # concatenate each view i to Ht
        H.append(InterviewAttention(z, Ht)) # get scaled Ht
        
    H = Concatenate()(H)
    H = Reshape(target_shape=[length, n_districts, 1])(H)
    H = Concatenate()([H, F])
    H = TemporalAttention(gru_h, H, length)
    
    H = Dense(fc_h, activation='relu')(H)
    H = Dropout(0.1)(H)
    H = Dense(fc_h, activation='relu')(H)
    H = Dropout(0.1)(H)
        
    y = Dense(n_steps * n_districts)(H)
    y = Reshape([n_steps, n_districts])(y)

    # A_train, A_poi_train, A_demo_train, A_road_train, A_traffic_train, node_features_train
    model = Model(inputs=[A_S, A_P, A_D, A_R, A_T, F], outputs=y)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber(delta=d))
    model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_val, y_val), callbacks=[es], verbose=0)
    return model

def MG_TAR_V(x_train, y_train, x_val, y_val, configs, length=12, n_steps=6, view='All'):
    tf.keras.backend.clear_session()
    
    _, _, _, _, _, node_features = x_train
    _, _, n_districts, n_features = node_features.shape
    gru_h, gcn_f, fc_h, n_layers, bn, d = configs

    A_S = Input(shape=[n_districts, n_districts]) # spatial closeness
    A_P = Input(shape=[n_districts, n_districts]) # functional similarity (POI)
    A_R = Input(shape=[n_districts, n_districts]) # road similarity
    A_D = Input(shape=[n_districts, n_districts]) # demographic similarity
    A_T = Input(shape=[length, n_districts, n_districts]) # traffic patterns

    F = Input(shape=[length, n_districts, n_features]) # node features
    
    H = [] # H_1 to H_T

    for t in range(length):
        # slice for each time step t
        Ft = Lambda(lambda f: f[:,t,:,:])(F)
        A_Tt = Lambda(lambda a: a[:,t,:,:])(A_T)

        X_S, X_P, X_R, X_D, X_Tt = Ft, Ft, Ft, Ft, Ft # input H_t0 time t layer 0
        H_S, H_P, H_R, H_D, H_Tt = X_S, X_P, X_R, X_D, X_Tt

        for i in range(n_layers): # using aggregation for each layer as in ST-MGCN ?
            if view == 'All' or view == 'OS':
                H_S = GCNConv(gcn_f)([H_S, A_S]) # GCN for Adjacency Matrix
                if (i + 1) % 2 == 0:
                    H_S = BatchNormalization()(H_S) if bn else H_S
                H_S = Activation('relu')(H_S)
                
            if view == 'All' or (view != 'P' and view != 'OS'):
                H_P = GCNConv(gcn_f)([H_P, A_P]) # GCN for POI Graph
                if (i + 1) % 2 == 0:
                    H_P = BatchNormalization()(H_P) if bn else H_P
                H_P = Activation('relu')(H_P)
                
            if view == 'All' or (view != 'D' and view != 'OS'):
                H_D = GCNConv(gcn_f)([H_D, A_D])  # GCN for Demographic Graph
                if (i + 1) % 2 == 0:
                    H_D = BatchNormalization()(H_D) if bn else H_D
                H_D = Activation('relu')(H_D)
            
            if view == 'All' or (view != 'R' and view != 'OS'):
                H_R = GCNConv(gcn_f)([H_R, A_R]) # GCN for Road Graph
                if (i + 1) % 2 == 0:
                    H_R = BatchNormalization()(H_R) if bn else H_R
                H_R = Activation('relu')(H_R)   
                
            if view == 'All' or (view != 'T' and view != 'OS'):
                H_Tt = GCNConv(gcn_f)([H_Tt, A_Tt]) # GCN for Traffic Patterns
                if (i + 1) % 2 == 0:
                    H_Tt = BatchNormalization()(H_Tt) if bn else H_Tt
                H_Tt = Activation('relu')(H_Tt) 

        
        if view == 'All':
            H_S = GCNConv(1, activation='relu')([H_S, A_S]) 
            H_P = GCNConv(1, activation='relu')([H_P, A_P])
            H_D = GCNConv(1, activation='relu')([H_D, A_D])
            H_R = GCNConv(1, activation='relu')([H_R, A_R])
            H_Tt = GCNConv(1, activation='relu')([H_Tt, A_Tt])

            # summarize each channel (i.e., view) into a scalar 
            z = Concatenate()([GlobalAvgPool()(H_S), GlobalAvgPool()(H_P), GlobalAvgPool()(H_D), GlobalAvgPool()(H_R), GlobalAvgPool()(H_Tt)]) # concatenate it into vector z
            Ht = Concatenate()([H_S, H_P, H_D, H_R, H_Tt]) # concatenate each view i to Ht
            H.append(InterviewAttention(z, Ht)) # get scaled Ht

        elif view == 'OS':
            H_S = GCNConv(1, activation='relu')([H_S, A_S]) 

            # summarize each channel (i.e., view) into a scalar 
            z = GlobalAvgPool()(H_S) # concatenate it into vector z
            Ht = H_S # concatenate each view i to Ht
            H.append(InterviewAttention(z, Ht)) # get scaled Ht

        elif view == 'S':
            H_P = GCNConv(1, activation='relu')([H_P, A_P])
            H_D = GCNConv(1, activation='relu')([H_D, A_D])
            H_R = GCNConv(1, activation='relu')([H_R, A_R])
            H_Tt = GCNConv(1, activation='relu')([H_Tt, A_Tt])

            # summarize each channel (i.e., view) into a scalar 
            z = Concatenate()([GlobalAvgPool()(H_P), GlobalAvgPool()(H_D), GlobalAvgPool()(H_R), GlobalAvgPool()(H_Tt)]) # concatenate it into vector z
            Ht = Concatenate()([H_P, H_D, H_R, H_Tt]) # concatenate each view i to Ht
            H.append(InterviewAttention(z, Ht)) # get scaled Ht

        elif view == 'P':
            H_S = GCNConv(1, activation='relu')([H_S, A_S]) 
            H_D = GCNConv(1, activation='relu')([H_D, A_D])
            H_R = GCNConv(1, activation='relu')([H_R, A_R])
            H_Tt = GCNConv(1, activation='relu')([H_Tt, A_Tt])

            # summarize each channel (i.e., view) into a scalar 
            z = Concatenate()([GlobalAvgPool()(H_S), GlobalAvgPool()(H_D), GlobalAvgPool()(H_R), GlobalAvgPool()(H_Tt)]) # concatenate it into vector z
            Ht = Concatenate()([H_S, H_D, H_R, H_Tt]) # concatenate each view i to Ht
            H.append(InterviewAttention(z, Ht)) # get scaled Ht

        elif view == 'R':
            H_S = GCNConv(1, activation='relu')([H_S, A_S]) 
            H_P = GCNConv(1, activation='relu')([H_P, A_P])
            H_D = GCNConv(1, activation='relu')([H_D, A_D])
            H_Tt = GCNConv(1, activation='relu')([H_Tt, A_Tt])

            # summarize each channel (i.e., view) into a scalar 
            z = Concatenate()([GlobalAvgPool()(H_S), GlobalAvgPool()(H_P), GlobalAvgPool()(H_D), GlobalAvgPool()(H_Tt)]) # concatenate it into vector z
            Ht = Concatenate()([H_S, H_P, H_D, H_Tt]) # concatenate each view i to Ht
            H.append(InterviewAttention(z, Ht)) # get scaled Ht

        elif view == 'D':
            H_S = GCNConv(1, activation='relu')([H_S, A_S]) 
            H_P = GCNConv(1, activation='relu')([H_P, A_P])
            H_R = GCNConv(1, activation='relu')([H_R, A_R])
            H_Tt = GCNConv(1, activation='relu')([H_Tt, A_Tt])

            # summarize each channel (i.e., view) into a scalar 
            z = Concatenate()([GlobalAvgPool()(H_S), GlobalAvgPool()(H_P), GlobalAvgPool()(H_R), GlobalAvgPool()(H_Tt)]) # concatenate it into vector z
            Ht = Concatenate()([H_S, H_P, H_R, H_Tt]) # concatenate each view i to Ht
            H.append(InterviewAttention(z, Ht)) # get scaled Ht

        elif view == 'T':
            H_S = GCNConv(1, activation='relu')([H_S, A_S]) 
            H_P = GCNConv(1, activation='relu')([H_P, A_P])
            H_D = GCNConv(1, activation='relu')([H_D, A_D])
            H_R = GCNConv(1, activation='relu')([H_R, A_R])

            # summarize each channel (i.e., view) into a scalar 
            z = Concatenate()([GlobalAvgPool()(H_S), GlobalAvgPool()(H_P), GlobalAvgPool()(H_D), GlobalAvgPool()(H_R)]) # concatenate it into vector z
            Ht = Concatenate()([H_S, H_P, H_D, H_R]) # concatenate each view i to Ht
            H.append(InterviewAttention(z, Ht)) # get scaled Ht

        
    H = Concatenate()(H)
    H = Reshape(target_shape=[length, n_districts, 1])(H)
    H = Concatenate()([H, F])
    H = TemporalAttention(gru_h, H, length)
    
    
    H = Dense(fc_h, activation='relu')(H)
    H = Dropout(0.1)(H)
    H = Dense(fc_h, activation='relu')(H)
    H = Dropout(0.1)(H)
        
    y = Dense(n_steps * n_districts)(H)
    y = Reshape([n_steps, n_districts])(y)

    # A_train, A_poi_train, A_demo_train, A_road_train, A_traffic_train, node_features_train
    model = Model(inputs=[A_S, A_P, A_D, A_R, A_T, F], outputs=y)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber(delta=d))
    model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_val, y_val), callbacks=[es], verbose=0)
    return model

def MG_TAR_A(x_train, y_train, x_val, y_val, configs, length=12, n_steps=6, attn='All'):
    tf.keras.backend.clear_session()
    
    _, _, _, _, _, node_features = x_train
    _, _, n_districts, n_features = node_features.shape
    gru_h, gcn_f, fc_h, n_layers, bn, d = configs

    A_S = Input(shape=[n_districts, n_districts]) # spatial closeness
    A_P = Input(shape=[n_districts, n_districts]) # functional similarity (POI)
    A_R = Input(shape=[n_districts, n_districts]) # road similarity
    A_D = Input(shape=[n_districts, n_districts]) # demographic similarity
    A_T = Input(shape=[length, n_districts, n_districts]) # traffic patterns

    F = Input(shape=[length, n_districts, n_features]) # node features
    
    H = [] # H_1 to H_T

    for t in range(length):
        # slice for each time step t
        Ft = Lambda(lambda f: f[:,t,:,:])(F)
        A_Tt = Lambda(lambda a: a[:,t,:,:])(A_T)

        X_S, X_P, X_R, X_D, X_Tt = Ft, Ft, Ft, Ft, Ft # input H_t0 time t layer 0
        H_S, H_P, H_R, H_D, H_Tt = X_S, X_P, X_R, X_D, X_Tt

        for i in range(n_layers): # using aggregation for each layer as in ST-MGCN ?
            H_S = GCNConv(gcn_f)([H_S, A_S]) # GCN for Adjacency Matrix
            if (i + 1) % 2 == 0:
                H_S = BatchNormalization()(H_S) if bn else H_S
            H_S = Activation('relu')(H_S)

            H_P = GCNConv(gcn_f)([H_P, A_P]) # GCN for POI Graph
            if (i + 1) % 2 == 0:
                H_P = BatchNormalization()(H_P) if bn else H_P
            H_P = Activation('relu')(H_P)

            H_D = GCNConv(gcn_f)([H_D, A_D])  # GCN for Demographic Graph
            if (i + 1) % 2 == 0:
                H_D = BatchNormalization()(H_D) if bn else H_D
            H_D = Activation('relu')(H_D)

            H_R = GCNConv(gcn_f)([H_R, A_R]) # GCN for Road Graph
            if (i + 1) % 2 == 0:
                H_R = BatchNormalization()(H_R) if bn else H_R
            H_R = Activation('relu')(H_R)   

            H_Tt = GCNConv(gcn_f)([H_Tt, A_Tt]) # GCN for Traffic Patterns
            if (i + 1) % 2 == 0:
                H_Tt = BatchNormalization()(H_Tt) if bn else H_Tt
            H_Tt = Activation('relu')(H_Tt) 

            
        H_S = GCNConv(1, activation='relu')([H_S, A_S]) 
        H_P = GCNConv(1, activation='relu')([H_P, A_P])
        H_D = GCNConv(1, activation='relu')([H_D, A_D])
        H_R = GCNConv(1, activation='relu')([H_R, A_R])
        H_Tt = GCNConv(1, activation='relu')([H_Tt, A_Tt])
        
        if attn == 'View':
            z = Concatenate()([GlobalAvgPool()(H_S), GlobalAvgPool()(H_P), GlobalAvgPool()(H_D), GlobalAvgPool()(H_R), GlobalAvgPool()(H_Tt)]) # concatenate it into vector z
            H.append(z)
        else:
            # summarize each channel (i.e., view) into a scalar 
            z = Concatenate()([GlobalAvgPool()(H_S), GlobalAvgPool()(H_P), GlobalAvgPool()(H_D), GlobalAvgPool()(H_R), GlobalAvgPool()(H_Tt)]) # concatenate it into vector z
            Ht = Concatenate()([H_S, H_P, H_D, H_R, H_Tt]) # concatenate each view i to Ht
            H.append(InterviewAttention(z, Ht)) # get scaled Ht

    if attn == 'Temp':
        H = Concatenate()(H)
    elif attn == 'View':
        H = Concatenate()(H)
        H = Reshape(target_shape=[length, 5, 1])(H)
        H = TemporalAttention(gru_h, H, length)
    else:
        H = Concatenate()(H)
        H = Reshape(target_shape=[length, n_districts, 1])(H)
        H = Concatenate()([H, F])
        H = TemporalAttention(gru_h, H, length)
    
    
    H = Dense(fc_h, activation='relu')(H)
    H = Dropout(0.1)(H)
    H = Dense(fc_h, activation='relu')(H)
    H = Dropout(0.1)(H)
        
    y = Dense(n_steps * n_districts)(H)
    y = Reshape([n_steps, n_districts])(y)

    # A_train, A_poi_train, A_demo_train, A_road_train, A_traffic_train, node_features_train
    model = Model(inputs=[A_S, A_P, A_D, A_R, A_T, F], outputs=y)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber(delta=d))
    model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_val, y_val), callbacks=[es], verbose=0)
    return model
