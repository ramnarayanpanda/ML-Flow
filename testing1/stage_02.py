import mlflow 
import pickle


def func():
    param_dct = {'seed':42, 'max_features':5}
    metric_dct = {'rec':88, 'f1':89}
    
    with open('./temp/param.p', 'rb') as fp:
        param_dct1 = pickle.load(fp)
    with open('./temp/metric.p', 'rb') as fp:
        metric_dct1 = pickle.load(fp)
        
    param_dct.update(param_dct1)
    metric_dct.update(metric_dct1)
    
    with open('./temp/param.p', 'wb') as fp:
        pickle.dump(param_dct, fp)
    with open('./temp/metric.p', 'wb') as fp:
        pickle.dump(metric_dct, fp)  
    
    
if __name__ == '__main__':
    func()