import mlflow 
import pickle


def func():
    param_dct = {'batch_size':2, 'no_of_iterations':5}
    metric_dct = {'acc':95, 'roc':93}
    
    with open('./temp/param.p', 'wb') as fp:
        pickle.dump(param_dct, fp)
    with open('./temp/metric.p', 'wb') as fp:
        pickle.dump(metric_dct, fp)  
    
    
if __name__ == '__main__':
    func()