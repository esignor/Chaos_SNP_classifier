## MODULE
import sys
sys.path.insert(1, 'CODE AND EXPERIMENTS/CGR-pcmer/')
import AQUACULTURE
from AQUACULTURE.module import *

## FUNCTIONS
from AQUACULTURE.functions_Net_AQUACULTURE import create_AlexNet, preprocessing, metrics, saveModel, plot_loss_accuracy, saveConfMatrixClassReport # RGB
from AQUACULTURE.shapleyValues_functions import shapleyImagePlot

#from AQUACULTURE.functions_Net_AQUACULTURE import  plot_loss, plot_accuracy, metrics, saveModel, plot_loss_accuracy, saveConfMatrixClassReport # GS
#from AQUACULTURE.functions_NetGS_AQUACULTURE import create_AlexNet, preprocessing # GS
if __name__ == '__main__':
    
      num = 123
      np.random.seed(num)
      os.environ['PYTHONHASHSEED'] = str(num)
      os.environ['TF_DETERMINISTIC_OPS'] = '1'
      os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
      tf.random.set_seed(num)
      tf.keras.utils.set_random_seed(num)

      ## setting parameters
      dataset_cgr = 'features-active50'
      type_encoder = "Grayscale"
      dataset_train = 'include-chr3/CGR/GS 4 points/' + dataset_cgr + '/0'
      dataset_test = 'include-chr3/CGR/GS 4 points/' + dataset_cgr +'/1'
      batch_size=15; epoch=120
      channel = "RGB"
      #channel = "GS"
      
      X_data, y_data, nb_classes = preprocessing('AlexNet', type_encoder, dataset_train)
      X_data = np.array(X_data)
      y_data = np.array(y_data)     
      X_data = X_data.reshape((-1, 227, 227, 3)) # for RGB channel
      #X_data = X_data.reshape((-1, 227, 227, 1)) # for GS channel
      X_data = X_data.astype('float32')
      print('nb_classes data', nb_classes)

      X_test, y_test, nb_classes = preprocessing('AlexNet', type_encoder, dataset_test)
      X_test = np.array(X_test)
      y_test = np.array(y_test)     
      X_test = X_test.reshape((-1, 227, 227, 3)) # for RGB channel
      #X_test = X_test.reshape((-1, 227, 227, 1)) # for GS channel
      X_test = X_test.astype('float32')
     
      print('nb_classes test', nb_classes)
      print('train shape: {}'.format(X_data.shape))
      print('train labels shape: {}'.format(y_data.shape))
      
      print('test shape: {}'.format(X_test.shape))
      print('test labels shape: {}'.format(y_test.shape))
      print('y test', y_test)
     
      skf_AlexNet = StratifiedKFold(n_splits=5,shuffle=True,random_state=20)
      tmp=1; model_AlexNet = Sequential()      

      start = time.time()
      n_task = 2
      for train_index, validation_index in skf_AlexNet.split(X_data, y_data):
          model_AlexNet = Sequential()
          X_train, X_validation = X_data[train_index], X_data[validation_index]
          y_train, y_validation = y_data[train_index], y_data[validation_index]
          print('Fold'+str(tmp)+':')   

          with ProcessPoolExecutor(n_task) as e:
            e.map(create_AlexNet, range(n_task))      
            model_AlexNet = create_AlexNet(model_AlexNet, (227,227,3), nb_classes)
            history=model_AlexNet.fit(X_train[:], y_train[:],
                  batch_size=batch_size,
                  epochs=epoch,
                  validation_data=(X_validation[:], y_validation[:]))
            print('Fold'+str(tmp)+'is finished')
      end = time.time() 
      
      val_acc = "Validation accuracy" + str((history.history['val_accuracy'])[-1]) 

      training_time  = "model training time of AlexNet Model with " + type_encoder + " encoder unit: " + str(end-start) + ' s'
      print(training_time)    
          
       # save the classification model as a file
      saveModel(dataset_train, model_AlexNet, 'AlexNet', type_encoder)    

      acc = plot_loss_accuracy(history, model_AlexNet, X_test, y_test, dataset_train, 'AlexNet', type_encoder)

      conf_matrix, class_report, y_predict  = metrics(X_test, y_test, model_AlexNet)
      print('\n', conf_matrix, '\n', class_report)

      # save the results of classification model
      saveConfMatrixClassReport('AlexNet', training_time, acc, conf_matrix, class_report, dataset_test, type_encoder)
      # Shapley Values
      #shapleyImagePlot(X_data, y_data, X_test, y_test, model_AlexNet, dataset_test, channel, dataset_cgr, y_predict)
