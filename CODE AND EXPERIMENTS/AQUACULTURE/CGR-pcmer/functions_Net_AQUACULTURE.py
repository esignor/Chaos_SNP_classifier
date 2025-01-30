import sys
sys.path.insert(1, 'CODE AND EXPERIMENTS/CGR-pcmer/')
import AQUACULTURE
from AQUACULTURE.module import *

def create_SimplerCNN(model, shape, nb_classes):
  model.add(layers.Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), activation="relu", input_shape=shape, padding = "same"))
  model.add(layers.MaxPool2D(pool_size=(2, 2)))
  model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding = "same"))
  model.add(layers.MaxPool2D(pool_size=(2, 2)))
  model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding = "same"))
  model.add(layers.MaxPool2D(pool_size=(2, 2)))
  model.add(layers.Flatten())
  model.add(layers.Dense(128, activation = "sigmoid"))
  model.add(layers.Dropout(0.25, seed=0))
  model.add(layers.Dense(1, activation = "sigmoid"))
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  return model

def create_ComplexCNN(model, shape, nb_classes):
  #model = Sequential()
  model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation="relu", input_shape=shape, padding = "same"))
  model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding = "same"))
  model.add(layers.MaxPool2D(pool_size=(2, 2)))
  model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding = "same"))
  model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding = "same"))
  model.add(layers.MaxPool2D(pool_size=(2, 2)))
  model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding = "same"))
  model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding = "same"))
  model.add(layers.MaxPool2D(pool_size=(2, 2)))
  model.add(layers.Flatten())
  model.add(layers.Dense(128, activation = "sigmoid"))
  model.add(layers.Dropout(0.25, seed=0))
  model.add(layers.Dense(1, activation = "sigmoid"))
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  return model
  
def create_AlexNet(model, shape, nb_classes):
  #model = Sequential()
  model.add(layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation="relu", input_shape= shape))
  model.add(layers.BatchNormalization())
  model.add(layers.MaxPool2D(pool_size=(3, 3), strides= (2, 2)))
  model.add(layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation="relu", padding="same"))
  model.add(layers.BatchNormalization())
  model.add(layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
  model.add(layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="same"))
  model.add(layers.BatchNormalization())
  model.add(layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="same"))
  model.add(layers.BatchNormalization())
  model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="same"))
  model.add(layers.BatchNormalization())
  model.add(layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
  model.add(layers.Flatten())
  model.add(layers.Dense(4096, activation="sigmoid"))
  model.add(layers.Dropout(0.5, seed=0))
  model.add(layers.Dense(4096, activation="sigmoid"))
  model.add(layers.Dropout(0.5, seed=0))
  model.add(layers.Dense(1, activation="sigmoid"))
  model.compile(loss='binary_crossentropy', optimizer=tf.optimizers.SGD(learning_rate=0.001), metrics=['accuracy'])
  return model


def convolutional_block(X, f, filters, stage, block, s = 2):
  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'
  F1, F2, F3 = filters
  X_shortcut = X
  X = layers.Conv2D(filters = F1, kernel_size = (1, 1), strides = (s,s), name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
  X = layers.BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
  X = layers.Activation('relu')(X)
  X = layers.Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
  X = layers.BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
  X = layers.Activation('relu')(X)
  X = layers.Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
  X = layers.BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)
  X_shortcut = layers.Conv2D(filters = F3, kernel_size = (1, 1), strides = (s,s), padding = 'valid', name = conv_name_base + '1', kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
  X_shortcut = layers.BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)
  X = layers.Add()([X_shortcut, X])
  X = layers.Activation('relu')(X)

  return X


def identity_block(X, f, filters, stage, block):

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    X = layers.Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = layers.BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = layers.Activation('relu')(X)

    X = layers.Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = layers.BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = layers.Activation('relu')(X)

    X = layers.Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = layers.BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Add shortcut value to main path
    X = layers.Add()([X_shortcut, X])
    X = layers.Activation('relu')(X)

    return X

def create_ResNet50(model, shape, nb_classes):
    X_input = layers.Input(shape)
    X = layers.ZeroPadding2D((3, 3))(X_input)
    X = layers.Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = layers.BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = layers.Activation('relu')(X)
    X = layers.MaxPooling2D((3, 3), strides=(2, 2))(X)
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')
    X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 3, block='a', s = 2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')
    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')
    X = convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block='a', s = 2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')
    X = layers.AveragePooling2D(pool_size=(2, 2),name='avg_pool')(X)
    X = layers.Flatten()(X)
    X = layers.Dense(1, activation='sigmoid', name='fc' + str(nb_classes), kernel_initializer = glorot_uniform(seed=0))(X)
    model = Model(inputs = X_input, outputs = X, name='ResNet50')
    model.compile(loss='binary_crossentropy', optimizer=tf.optimizers.SGD(learning_rate=0.001), metrics=['accuracy'])
    return model


def create_ResNet101(model, shape, nb_classes):
    X_input = layers.Input(shape)
    X = layers.ZeroPadding2D((3, 3))(X_input)
    X = layers.Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = layers.BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = layers.Activation('relu')(X)
    X = layers.MaxPooling2D((3, 3), strides=(2, 2))(X)

    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='d')

    X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 3, block='a', s = 2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='e')

    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='g')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='h')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='i')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='l')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='m')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='n')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='o')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='p')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='q')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='r')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='s')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='t')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='u')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='v')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='z')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='bb')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='bc')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='bd')


    X = convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block='a', s = 2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='d')

    X = layers.AveragePooling2D(pool_size=(7, 7),name='avg_pool')(X)
    X = layers.Flatten()(X)
    X = layers.Dense(1, activation='sigmoid', name='fc' + str(nb_classes), kernel_initializer = glorot_uniform(seed=0))(X)
    model = Model(inputs = X_input, outputs = X, name='ResNet101')
    model.compile(loss='binary_crossentropy', optimizer=tf.optimizers.SGD(learning_rate=0.001), metrics=['accuracy'])
    return model


  
def preprocessing(type_arch, type_encoder, dataset):

## for the resize method bicubic resampling is used (default value): there is a fixed
# number of source image pixels for each target pixel (i.e., 4*4 for bicubic) where 
# affine transformations are applied.

    X = []; y = []
    if type_encoder == "Grayscale":
      path_main = 'CODE AND EXPERIMENTS/CGR-pcmer/AQUACULTURE/OUTGrayScaleCGR/Aquaculture/' + dataset
    os.chdir(path_main)
    dirs = filter(os.path.isdir, os.listdir(os.curdir))
    for dir in dirs:
        path_to_subdir = str(dir)
        for im_path in os.listdir(path_to_subdir):
            im_frame = Image.open(path_to_subdir + '/' + im_path)
            if type_arch == "AlexNet":
              ## resize image for AlexNet
              single_channel = im_frame.resize((227, 227))
              np_frame = np.concatenate([single_channel, single_channel, single_channel], axis = -1)

            elif type_arch == "ResNet":
              ## resize image for ResNet
              single_channel = im_frame.resize((224, 224))
              np_frame = np.concatenate([single_channel, single_channel, single_channel], axis = -1)

            
            elif type_arch == "CNN":
              ## resize image for CNNs
              single_channel = im_frame.resize((256, 256))
              np_frame = np.concatenate([single_channel], axis = -1)
            
            y.append(path_to_subdir.split('.')[0]); X.append(np_frame)

    os.chdir('../../../../../../../../../../')
    unique = list(dict.fromkeys(y))
    dct = {}
    cnt = 0
    for lab in unique:
        dct[str(lab)] = cnt
        cnt += 1

    nb_classes = len(dct)
    new_label = []
    for l in y:
        new_label.append(dct[l])

    y = new_label
    return X, y, nb_classes
    

def plot_loss(history, dataset, model_net, type_encoder):
  directory = 'CODE AND EXPERIMENTS/CGR-pcmer/AQUACULTURE/OUTGrayScaleCGR/Aquaculture/' + dataset + ' Results/'
  if not os.path.exists(directory):
    os.makedirs(directory)

  plt.figure(figsize=(10,6))
  plt.plot(history.epoch,history.history['loss'], label = "Training loss" )
  plt.plot(history.epoch,history.history['val_loss'], label = "Validation loss")
  plt.title('loss')
  plt.legend(loc="lower right")
  plt.savefig(directory + '/' + type_encoder + " Training-Validation Loss " + model_net)
  plt.clf(); plt.close()

def plot_accuracy(history, dataset, model_net, type_encoder):
  directory='CODE AND EXPERIMENTS/CGR-pcmer/AQUACULTURE/OUTGrayScaleCGR/Aquaculture/' + dataset + ' Results/'
  if not os.path.exists(directory):
    os.makedirs(directory)

  plt.figure(figsize=(10,6))
  plt.plot(history.epoch,history.history['accuracy'], label = "Training accuracy")
  plt.plot(history.epoch,history.history['val_accuracy'], label = "Validation accuracy")
  plt.title('accuracy')
  plt.legend(loc="lower right")
  plt.savefig(directory + '/' + type_encoder + " Training-Validation Accuracy " + model_net)
  plt.clf(); plt.close()
  
def metrics(X_test, y_test, model_net):
  y_predict = model_net.predict(X_test)
  y_maxPredict = np.arange(len(y_test))
  print('pred', y_predict)
  index = 0
  for a in y_predict:
    if np.all(a > 0.5): mortality = 1
    else: mortality = 0
    np.put(y_maxPredict,[index],[mortality])
    index += 1

  print(y_maxPredict)
  # check results
  return confusion_matrix(y_test, y_maxPredict), classification_report(y_test, y_maxPredict, digits=4), y_maxPredict
  
  
def saveModel(dataset, model, net, type_encoder):
  directory = 'CODE AND EXPERIMENTS/CGR-pcmer/AQUACULTURE/OUTGrayScaleCGR/Aquaculture/' + dataset + ' Models/'
  if not os.path.exists(directory):
    os.makedirs(directory)

  model_file = directory + type_encoder + "model " + net + ".keras"

  #with open(model_file, 'wb') as file:
    #pickle.dump(model, file) # dump and pickle for to store the object data to the file
    #file.close()
    
  model.save(model_file)
  print("Save Model!")
    
def plot_loss_accuracy(history, model, X_test, y_test, dataset, model_net, type_encoder):
  plot_loss(history, dataset, model_net, type_encoder)
  plot_accuracy(history, dataset, model_net, type_encoder)

  scores = model.evaluate(X_test, y_test, verbose=2)
  acc = "%s: %.2f%%" % (model.metrics_names[1], scores[1]*100)
  return acc
  
def saveConfMatrixClassReport(net, acc, training_time, conf_matrix, class_report, dataset, type_encoder):
  directory = 'CODE AND EXPERIMENTS/CGR-pcmer/AQUACULTURE/OUTGrayScaleCGR/Aquaculture/' + dataset + ' Results/'
  if not os.path.exists(directory):
    os.makedirs(directory)

  results_model_file = directory + type_encoder + "results " + net + ".txt"
  print("Save accuracy and classification report!")

  with open(results_model_file, 'w') as file:
    file.write('confusion matrix: \n' + str(conf_matrix) + '\n\n')
    file.write('classification report: \n' + str(class_report) + '\n')
    file.write(str(acc) + '\n')
    file.write(str(training_time))
    file.close()
