## MODULE
import sys
sys.path.insert(1, 'CODE AND EXPERIMENTS/CGR-pcmer/')
import AQUACULTURE
from AQUACULTURE.module import *

## FUNCTIONS
from AQUACULTURE.functions_Net_AQUACULTURE import create_ResNet50, preprocessing, plot_loss, plot_accuracy, metrics, saveModel, plot_loss_accuracy, saveConfMatrixClassReport

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
      dataset_C0 = 'include-chr3/CGR/GS 4 points/' + dataset_cgr + '/0'
      dataset_C1 = 'include-chr3/CGR/GS 4 points/' + dataset_cgr +'/1'
      batch_size=15; epoch=90

      X_dataC0, y_dataC0, nb_classes = preprocessing('ResNet', type_encoder, dataset_C0)
      X_dataC0 = np.array(X_dataC0)
      y_dataC0 = np.array(y_dataC0)     
      X_dataC0 = X_dataC0.reshape((-1, 224, 224, 3));    
      X_dataC0 = X_dataC0.astype('float32')
      print('nb_classes data', nb_classes)

      X_dataC1, y_dataC1, nb_classes = preprocessing('ResNet', type_encoder, dataset_C1)
      X_dataC1 = np.array(X_dataC1)
      y_dataC1 = np.array(y_dataC1)     
      X_dataC1 = X_dataC1.reshape((-1, 224, 224, 3));    
      X_dataC1 = X_dataC1.astype('float32')
      print('nb_classes test', nb_classes)
      
      
      X_data = np.concatenate((X_dataC0, X_dataC1))
      y_data = np.concatenate((y_dataC0, y_dataC1))
      
      
      X = X_data; y = y_data
      # Indexing for training and validation sets (80)
      # Indexing for testing set (20)
      X_data, X_test, y_data, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify = y) 
      
      
      print('train shape: {}'.format(X_data.shape))
      print('train labels shape: {}'.format(y_data.shape))
      
      print('test shape: {}'.format(X_test.shape))
      print('test labels shape: {}'.format(y_test.shape))

      skf_ResNet50 = StratifiedKFold(n_splits=5,shuffle=True,random_state=20)
      tmp=1; model_ResNet50 = Sequential()      

      start = time.time()
      n_task = 2
      for train_index, validation_index in skf_ResNet50.split(X_data, y_data):
          model_ResNet50 = Sequential()
          X_train, X_validation = X_data[train_index], X_data[validation_index]
          y_train, y_validation = y_data[train_index], y_data[validation_index]
          print('Fold'+str(tmp)+':')   

          with ProcessPoolExecutor(n_task) as e:
            e.map(create_ResNet50, range(n_task))      
            model_ResNet50 = create_ResNet50(model_ResNet50, (224,224,3), nb_classes)
            history=model_ResNet50.fit(X_train[:], y_train[:],
                  batch_size=batch_size,
                  epochs=epoch,
                  validation_data=(X_validation[:], y_validation[:]))
            print('Fold'+str(tmp)+'is finished')
      end = time.time()  

      val_acc = "Validation accuracy" + str((history.history['val_accuracy'])[-1])

      training_time  = "model training time of ResNet50 Model with " + type_encoder + " encoder unit: " + str(end-start) + ' s'
      print(training_time)    
         
      # save the classification model as a file
      saveModel(dataset_C0, model_ResNet50, 'ResNet50', type_encoder + 'ResNet50' + str(batch_size) + str(epoch) + 'train80-test20')    

      acc = plot_loss_accuracy(history, model_ResNet50, X_test, y_test, dataset_C0, 'ResNet50', type_encoder + 'ResNet50' + str(batch_size) + str(epoch) + 'train80-test20')

      conf_matrix, class_report = metrics(X_test, y_test, model_ResNet50)
      print('\n', conf_matrix, '\n', class_report)

      # save the results of classification model
      saveConfMatrixClassReport('ResNet50', training_time, acc, conf_matrix, class_report, dataset_C0, type_encoder + 'ResNet50' + str(batch_size) + str(epoch) + 'train80-test20')
