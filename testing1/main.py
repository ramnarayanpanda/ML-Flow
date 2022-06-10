import mlflow 
import pickle



def func():
    data = {}
    with open('./temp/param.p', 'wb') as fp:
        pickle.dump(data, fp)
    with open('./temp/metric.p', 'wb') as fp:
        pickle.dump(data, fp)
        
    with mlflow.start_run() as active_run:
        mlflow.run(".", "stage_01", use_conda=False)
        mlflow.run(".", "stage_02", use_conda=False)  
        
        with open('./temp/param.p', 'rb') as fp:
            param_dct = pickle.load(fp)
        with open('./temp/metric.p', 'rb') as fp:
            metric_dct = pickle.load(fp)
            
        mlflow.log_metrics(metric_dct)
        mlflow.log_params(param_dct)  


if __name__ == '__main__':
    func()  