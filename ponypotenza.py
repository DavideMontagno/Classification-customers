import pandas as pd
import numpy as npp
import matplotlib.pyplot as plt
df = pd.read_csv("./dataset/CL_over_Normalized-dataset.csv",sep='\t',decimal=",",index_col=0)

df.head()


from sklearn.model_selection import train_test_split

label = df.pop('class')
train_set, test_set, train_label, test_label = train_test_split(df, label, stratify =label, test_size=0.20)


from itertools import product
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.layers import Dense, Dropout, InputLayer


from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import utils

encoder = LabelEncoder()
encoder.fit(train_label)
encoded_Y = encoder.transform(train_label)
train_label = utils.to_categorical(encoded_Y)

encoder = LabelEncoder()
encoder.fit(test_label)
encoded_Y = encoder.transform(test_label)
test_label = utils.to_categorical(encoded_Y)


#GRID SEARCH
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tr_epochs=1000
grid_params = {
                'hidden_layers': [1,2,3],
                'hidden_units': [16,32,64,128],
                'act_funct': ['relu', 'tanh'],
                'learning_rate': [1e-3,1e-4,1e-6, 1e-5],
                'optimizer': [Adam]
            }
keys, values = zip(*grid_params.items())
params_list = [dict(zip(keys, v)) for v in product(*values)]




dict_model={}
for idx, params in enumerate(params_list):
    print("**********" + str(idx) + "*************")
    
    #Parametri
    hidden_layers = params['hidden_layers']
    hidden_units = params['hidden_units']
    act_funct = params['act_funct']
    learning_rate = params['learning_rate']
    optimizer = params['optimizer']

    #Creazione del modello
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(1, 8)))

    for i in range(0, hidden_layers):
        model.add(Dense(hidden_units, activation = act_funct))
        model.add(tf.keras.layers.Dropout(0.2))

    model.add(Dense(3, activation = 'softmax')) #Inserire softmax di 3 neuroni

    model.compile(optimizer=optimizer(learning_rate=learning_rate),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    # Allenamento
    #y_train = np.asarray(train_label).astype('float32').reshape((-1,1))
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode="auto")
    history = model.fit(train_set, train_label,epochs=tr_epochs,validation_split=0.2,callbacks=[callback])

    '''#Plot
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    
    plt.figure()
    plt.plot(epochs, acc, 'b', label='Training Acc')
    plt.plot(epochs, val_acc, 'r', label='Validation Acc')
    plt.title("LR: "+str(learning_rate)+" ACT: "+act_funct+" LAYER: "+ str(hidden_layers)+ " UNITS: "+ str(hidden_units))


    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.legend()
    plt.savefig("./nn_grid/plt"+str(idx)+".png")'''


    val_acc = history.history['val_accuracy']
    # Track model
    dict_model[str(params)]=val_acc[-1]


print("Grid Search: ")
[print(dict_model[key]," ",key)for key in dict_model]
print("\nBest model: ")
print(max(dict_model, key=dict_model.get))
print("Val_Accuracy: ",dict_model[max(dict_model, key=dict_model.get)])


