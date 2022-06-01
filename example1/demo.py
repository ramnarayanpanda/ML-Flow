import os 
import mlflow 
import argparse
import time 


# you can log metric, parameter, artifact 

def evaluate(arg1, arg2):
    return (arg1 * 2) + (arg2 * 2)


def main(arg1, arg2):
    
    with mlflow.start_run():
        mlflow.log_param('param1', arg1)
        mlflow.log_param('param2', arg2)
        
        metric = evaluate(arg1, arg2)
        mlflow.log_metric('some_metric', metric)
        
        os.makedirs('temp', exist_ok=True) 
        with open('temp/sample.txt', 'w') as f: 
            f.write(time.asctime())
        mlflow.log_artifacts('temp')
  

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--param1', '-p1', type=int, default=2)
    args.add_argument('--param2', '-p2', type=int, default=5)
    parsed_args = args.parse_args()
    
    main(parsed_args.param1, parsed_args.param2)