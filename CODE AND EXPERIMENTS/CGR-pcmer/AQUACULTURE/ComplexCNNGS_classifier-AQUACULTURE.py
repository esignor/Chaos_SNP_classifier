## MODULE
import sys
sys.path.insert(1, 'CODE AND EXPERIMENTS/CGR-pcmer/')
import AQUACULTURE
from AQUACULTURE.module import *

## FUNCTIONS
from AQUACULTURE.functions_Net_AQUACULTURE import create_ComplexCNN, preprocessing, plot_loss, plot_accuracy, metrics, saveModel, plot_loss_accuracy, saveConfMatrixClassReport

if __name__ == '__main__':
    
      ## setting parameters
      dataset_cgr = 'features-active50'
      type_encoder = "Grayscale"
      dataset_train = 'include-chr3/CGR/GS 4 points/' + dataset_cgr + '/0'
      dataset_test = 'include-chr3/CGR/GS 4 points/' + dataset_cgr +'/1'
      batch_size=150; epoch=1

      X_data, y_data, nb_classes = preprocessing('CNN', type_encoder, dataset_train)
      X_data = np.array(X_data)
      y_data = np.array(y_data)     
      X_data = X_data.reshape((-1, 256, 256, 1));    
      X_data = X_data.astype('float32')
      print('nb_classes data', nb_classes)
      shape = X_data.shape[1:]

      X_test, y_test, nb_classes = preprocessing('CNN', type_encoder, dataset_test)
      X_test = np.array(X_test)
      y_test = np.array(y_test)     
      X_test = X_test.reshape((-1, 256, 256, 1));    
      X_test = X_test.astype('float32')
      print('nb_classes test', nb_classes)
      
      print('train shape: {}'.format(X_data.shape))
      print('train labels shape: {}'.format(y_data.shape))
      
      print('test shape: {}'.format(X_test.shape))
      print('test labels shape: {}'.format(y_test.shape))

      skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=20)
      tmp=1; model_cnn = Sequential()

      start = time.time()
      for train_index, validation_index in skf.split(X_data, y_data):
        model_cnn = Sequential()
        X_train, X_validation = X_data[train_index], X_data[validation_index]
        y_train, y_validation = y_data[train_index], y_data[validation_index]
        print('Fold'+str(tmp)+':')
        model_cnn = create_ComplexCNN(model_cnn, shape, nb_classes)
        history=model_cnn.fit(X_train[:], y_train[:],
              batch_size=batch_size,
              epochs=epoch,
              validation_data=(X_validation[:], y_validation[:]))
        print('Fold'+str(tmp)+'is finished')
      end = time.time()

      val_acc = "Validation accuracy" + str((history.history['val_accuracy'])[-1])

      training_time  = "model training time of Complex CNN Model with " + type_encoder + " encoder unit: " + str(end-start) + ' s'
      print(training_time)    
         
      # save the classification model as a file
      saveModel(dataset_train, model_cnn, 'Complex CNN', type_encoder)    

      acc = plot_loss_accuracy(history, model_cnn, X_test, y_test, dataset_train, 'Complex CNN', type_encoder)

      conf_matrix, class_report, y_predict = metrics(X_test, y_test, model_cnn)
      print('\n', conf_matrix, '\n', class_report)

      # save the results of classification model
      saveConfMatrixClassReport('Complex CNN', training_time, acc, conf_matrix, class_report, dataset_test, type_encoder)
