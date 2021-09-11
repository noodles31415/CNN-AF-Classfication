import cnn_model as cnn
import Eval_model
import pickle
import numpy as np

def load_data(file_path):
    file = pickle.load(open(file_path, 'rb'))
    x_data = []
    y_data = []
    for y, x in file:
        x_data.append(x)
        y_data.append(x)

    return np.array(x_data).astype('float32'), np.array(y_data).astype('int8')


if __name__ == "__main__":
    
    #set file path
    train_med_path = './train_med_amp.pk1'
    test_med_path = './test_med_amp.pk1'

    
    # prepare median wave input
    x_med, y_med = load_data(train_med_path)
    x_med_test, y_med_test = load_data(test_med_path)
    x_med, y_med, x_med_test, y_med_test, y_med_true = cnn.InputPreprocess(x_med, y_med, x_med_test, y_med_test)


    #training data
    md_med, his_med = cnn.get_1DCNN(x_med, y_med, x_med_test, y_med_test, name='med') #1D-CNN

    

