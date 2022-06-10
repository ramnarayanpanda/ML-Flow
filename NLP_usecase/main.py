import mlflow 
import argparse 


def main(is_training=True):
    with mlflow.start_run() as run: 
        if is_training:
            print("######### Training #########")
            # We already have the data downloaded to data folder so dont run this stage
            # mlflow.run(".", "stage_01", use_conda=False)  
            mlflow.run(".", "stage_02", use_conda=False) 
            mlflow.run(".", "stage_03", use_conda=False) 
            mlflow.run(".", "stage_04", use_conda=False) 
            mlflow.run(".", "stage_05", use_conda=False) 
        else: 
            print("########## Evaluate ###########")
            mlflow.run(".", "stage_05", use_conda=False) 
            
    

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--training', '-t', type=int, default=1)  
    parsed_args = args.parse_args() 
    main(is_training=bool(parsed_args.training))         