## installing amulet after activating virtual env
pip install -U amlt --index-url https://msrpypi.azurewebsites.net/stable/leloojoo

## creating a project
amlt project create myproject phyagistudent

## identifying the storage account to amulet
amlt cred storage set phyagistudent
## then it asks for a key, which should be obtained from the azure storage expoloer > phyagistudnet 

## see targets for the singularity service
amlt target info sing

## starting an experiment
amlt run sing_simple.yaml simple_pytorch_experiment

## logging the task
amlt log -f simple_pytorch_experiment :0
amlt status simple_pytorch_experiment

