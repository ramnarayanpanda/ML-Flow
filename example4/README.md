This folder contains example of multi step workflow.
For one better example go to:  
https://github.com/mlflow/mlflow/tree/master/examples/multistep_workflow


Note:
In the MLproject file  we have diff stages such as main, stage_01, stage_02, stage_03.
However not all the stages will run when you do  "mlflow run . -P training=0 --no-conda" 
Only the main stage will run, other stages will not run. 
TO hack this you can define these stages inside MLproject file, but run inside main.py file using  "mlflow.run(".", "stage_02", use_conda=False)"


"mlflow run . -P training=0 --no-conda": We did not specify any env so it will just run main env.
"mlflow run . -e stage_01": This will run any particualr stage only


Important thing to note here is that how we pass parameters from main stage to main file, captured it using argparse.

