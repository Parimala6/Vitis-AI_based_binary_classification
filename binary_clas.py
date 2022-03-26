import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
#tf.compat.v1.enable_eager_execution() #enable for freeze_graph 
import numpy as np
from tensorflow import keras
import os, random
import cv2
from PIL import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.python.tools import freeze_graph
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.python.tools import optimize_for_inference_lib
#from tensorflow_model_optimization.quantization.keras import vitis_quantize
#import matplotlib.pyplot as plt


def preprocess(directory_path='data/mrlEyes_2018_01'):
    '''
    img_dir = os.listdir(directory_path)
    total = len(img_dir)
    count = 0
    os.system('rm -rf data/Train data/Test')
    os.system('mkdir data/Train data/Test')
    for dir in img_dir:
        dir = os.path.join(directory_path, dir)
        if count <= int(total/2)+1:
            train_list = os.listdir(dir)
            for train_img in train_list:
                os.system('cp ' + dir + '/' + train_img + ' data/Train/')
        else:
            test_list = os.listdir(dir)
            for test_img in test_list:
                os.system('cp ' + dir + '/' + test_img + ' data/Test/')    
        count = count + 1
    
    train_img_list = os.listdir('data/Train')
    os.system('rm -rf data/Train/Open data/Train/Close')
    os.system('mkdir data/Train/Open')
    os.system('mkdir data/Train/Close')
    for train_img in train_img_list:
        try:
            if(int(train_img.split('_')[4]) == 0):
                os.system('mv data/Train/' + train_img + ' data/Train/Close')
            else:
                os.system('mv data/Train/' + train_img + ' data/Train/Open')
        except IndexError:
            break
    
    test_img_list = os.listdir('data/Test')
    os.system('rm -rf data/Test/Open data/Test/Close')
    os.system('mkdir data/Test/Open')
    os.system('mkdir data/Test/Close')
    for test_img in test_img_list:
        try:
            if(int(test_img.split('_')[4]) == 0):
                os.system('mv data/Test/' + test_img + ' data/Test/Close')
            else:
                os.system('mv data/Test/' + test_img + ' data/Test/Open')
        except IndexError:
            break
    '''
    train = ImageDataGenerator(rescale=1/255, fill_mode='reflect', shear_range=0.2, width_shift_range=0.2, height_shift_range=0.2)        
    test = ImageDataGenerator(rescale=1/255)

    train_dataset = train.flow_from_directory("data/Train/", target_size=(150,150), batch_size = 32, class_mode = 'binary') #,color_mode='grayscale'
    test_dataset = test.flow_from_directory("data/Test/", target_size=(150,150), batch_size = 32, class_mode = 'binary') #color_mode='grayscale'
    print(test_dataset.class_indices)
    
    return train_dataset, test_dataset
    
      
def classifier_model(train_dataset, test_dataset):
    model = keras.Sequential()

    # Convolutional layer and maxpool layer 1
    model.add(keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
    model.add(keras.layers.MaxPool2D(2,2))

    # Convolutional layer and maxpool layer 2
    model.add(keras.layers.Conv2D(64,(3,3),activation='relu'))
    model.add(keras.layers.MaxPool2D(2,2))

    # Convolutional layer and maxpool layer 3
    model.add(keras.layers.Conv2D(128,(3,3),activation='relu'))
    model.add(keras.layers.MaxPool2D(2,2))

    # Convolutional layer and maxpool layer 4
    model.add(keras.layers.Conv2D(128,(3,3),activation='relu'))
    model.add(keras.layers.MaxPool2D(2,2))
    #model.add(keras.layers.Dropout(0.4))

    # This layer flattens the resulting image array to 1D array
    model.add(keras.layers.Flatten())

    # Hidden layer with 1024 neurons and Rectified Linear Unit activation function 
    model.add(keras.layers.Dense(1024,activation='relu'))
    
    # Hidden layer with 512 neurons and Rectified Linear Unit activation function 
    model.add(keras.layers.Dense(512,activation='relu'))
    #model.add(keras.layers.Dropout(0.4))

    # Output layer with single neuron which gives 0 for Close or 1 for Open 
    #Here we use sigmoid activation function which makes our model output to lie between 0 and 1
    model.add(keras.layers.Dense(1,activation='sigmoid'))

    return model
 
 
def train(train_dataset, test_dataset):
    
    with tf.Graph().as_default():
        model = classifier_model(train_dataset, test_dataset)

        #checkpoint
        filepath="weights.best.hdf5"
        checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
            
        model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

        model.fit_generator(train_dataset, 
                        steps_per_epoch = train_dataset.samples//train_dataset.batch_size, 
                        epochs = 10, 
                        validation_data = test_dataset, 
                        validation_steps=test_dataset.samples//test_dataset.batch_size, 
                        callbacks=callbacks_list, verbose=0)    
        print(model.summary()) 
            
        scores = model.evaluate(test_dataset, batch_size=32)
        print('Loss: %.3f' % scores[0])
        print('Accuracy: %.3f' % scores[1])
    
        with tf.compat.v1.Session() as sess:            
            init = tf.compat.v1.global_variables_initializer()
            sess.run(init)
            saver = tf.compat.v1.train.Saver()
            saver.save(sess,'./tensorflowModel.ckpt')
            tf.io.write_graph(sess.graph.as_graph_def(), '.', 'tensorflowModel.pbtxt', as_text=True)   
    
    # save weights, model architecture & optimizer to an HDF5 format file
    os.system('rm -rf saved_model')
    os.system('mkdir saved_model')
    model.save('saved_model/classification_model.h5')


def freeze_graph():
    model  = keras.models.load_model('saved_model/classification_model.h5')

    # Convert Keras model to ConcreteFunction
    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(
    tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype, name="conv2d_input:0"))
    
    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()
    layers = [op.name for op in frozen_func.graph.get_operations()]

    print("Frozen model layers: ")
    for layer in layers:
        print(layer)

    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)
    
    # Save frozen graph from frozen ConcreteFunction to hard drive
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir="./frozen_models",
                  name="frozen_graph.pb",
                  as_text=False)
    return
    

def optimize_graph():
    inputGraph = tf.GraphDef()
    with tf.gfile.Open('frozen_models/frozen_graph.pb', "rb") as f:
        data2read = f.read()
        inputGraph.ParseFromString(data2read)
  
    outputGraph = optimize_for_inference_lib.optimize_for_inference(
              inputGraph,
              ["placeholder"], # an array of the input node(s)
              ["sequential/dense_2/Sigmoid"], # an array of output nodes
              tf.int32.as_datatype_enum)

    # Save the optimized graph'test.pb'
    f = tf.gfile.FastGFile('frozen_models/OptimizedGraph.pb', "w")
    f.write(outputGraph.SerializeToString()) 

def wrap_frozen_graph(graph_def, inputs, outputs):
    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")

    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph

    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))

def evaluate(test_dataset):
    # Load frozen graph using TensorFlow 1.x functions
    with tf.io.gfile.GFile("./frozen_models/frozen_graph.pb", "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        loaded = graph_def.ParseFromString(f.read())

    # Wrap frozen graph to ConcreteFunctions
    frozen_func = wrap_frozen_graph(graph_def=graph_def,
                                    inputs=["placeholder"],
                                    outputs=["sequential/dense_2/Sigmoid"])

    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)

    # Get predictions for test images
    frozen_graph_predictions = frozen_func(x=tf.constant(test_dataset))[0]

    # Print the prediction for the first image
    print("Example TensorFlow frozen graph prediction reference:")
    print(frozen_graph_predictions[0].numpy())

    
def test():   
    model  = keras.models.load_model('saved_model/classification_model.h5')
    #inp = model.input
    #output = model.output
    #print(inp, output) 
    image_list = os.listdir('data/TestImages/')
    for img in image_list:
    	path = 'data/TestImages/' + img
    	img = image.load_img(path,target_size=(150,150))  
    	plt.imshow(img)
    	
    	Y = image.img_to_array(img)
    	X = np.expand_dims(Y,axis=0)
    	val = model.predict(X)
    	val = int(val[0][0])
    	print('value = ', val)
    	if val == 1: 
    		plt.xlabel("Open",fontsize=20) 
    	elif val == 0: 
    		plt.xlabel("Close",fontsize=20)
    	plt.show()

def main():
    train_dataset, test_dataset = preprocess()
    train(train_dataset, test_dataset)   
    #freeze_graph()
    #optimize_graph()
    #evaluate(test_dataset)
    #test()    
    
        
if __name__ == '__main__':
    main()                              
