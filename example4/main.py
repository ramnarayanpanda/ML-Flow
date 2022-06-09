import mlflow 
import argparse


def main(is_training=True): 
    print(f"\n\n>>>>>>>>>>{is_training}")
    with mlflow.start_run() as run: 
        mlflow.run(".", "stage_01", use_conda=False)
        mlflow.run(".", "stage_02", use_conda=False)
        
        # you can also ask the user if you want to run a stage or not
        # practical usage of this would be, let's say you are doing training, 
        # then you can run some of the workflows or for testing you can run some other pipelines
        if is_training==True:
            print("\n\n>>>>>>> Running stage3")
            mlflow.run(".", "stage_03", use_conda=False) 


if __name__=='__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--training', '-t', type=int, default=1)  
    parsed_args = args.parse_args() 
    print(f"\n\n>>>>>>>>>Here we are {parsed_args.training}  {type(parsed_args.training)} {bool(parsed_args.training)}") 
    main(is_training=bool(parsed_args.training)) 
    