from azureml.core import Workspace, Experiment
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core import ScriptRunConfig
from azureml.widgets import RunDetails
from azureml.train.sklearn import SKLearn
from sklearn.model_selection import train_test_split
from azureml.train.hyperdrive.run import PrimaryMetricGoal
from azureml.train.hyperdrive.policy import BanditPolicy
from azureml.train.hyperdrive.sampling import RandomParameterSampling
from azureml.train.hyperdrive.runconfig import HyperDriveConfig
from azureml.train.hyperdrive.parameter_expressions import uniform, choice
from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.train.automl import AutoMLConfig
from train import clean_data
from pandas_profiling import ProfileReport
import pandas as pd
import os
import shutil

#ws = Workspace.get(name="udacity-project")
ws = Workspace.from_config()
ws.write_config(path='.azureml')
experiment_name='udacity-project'
exp = Experiment(workspace=ws, name=experiment_name)

print('Workspace name: ' + ws.name, 
      'Azure region: ' + ws.location, 
      'Subscription id: ' + ws.subscription_id, 
      'Resource group: ' + ws.resource_group, sep = '\n')

run = exp.start_logging()

cpu_cluster_name = "OptimizePipe"
try:
    cpu_cluster = ComputeTarget(workspace=ws, name=cpu_cluster_name)
    print('A cluster with the same name already exists. If you are trying to create a new one please use a new cluster name')
except ComputeTargetException:
    print('Creating a new compute target...')
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_D2_V2',max_nodes=4,identity_type="SystemAssigned")
    cpu_cluster = ComputeTarget.create(ws, cpu_cluster_name, compute_config)
cpu_cluster.wait_for_completion(show_output=True)
# Get a detailed status for the current cluster. 
print(cpu_cluster.get_status().serialize())

compute_targets = ws.compute_targets
for name, ct in compute_targets.items():
    print(name, ct.type, ct.provisioning_state)

url_path = "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"
#Input Data
ds = TabularDatasetFactory.from_delimited_files(url_path, infer_column_types=True, separator=',', header=True, encoding='utf8')
x, y = clean_data(ds)
y_df=pd.DataFrame(y,columns=['y'])
data = pd.concat([x,y_df],axis=1)

training_data,validation_data = train_test_split(data,test_size = 0.3,random_state = 42,shuffle=True)

#convert the training dataset to a CSV file and store it under the training folder
training_data.to_csv('./training_data.csv')
#Create an experiment for the AutoML testing script
exp = Experiment(workspace=ws, name="AutoML-ModelTesting")
datastore = ws.get_default_datastore()
#Create a new folder 'data' and store training dataset into it using datastore
if "data" not in os.listdir():
    os.mkdir("./data")
# Get the dataset from the data folder
training_dataset = TabularDatasetFactory.from_delimited_files(path=[(('./training_data.csv'))])
# Set parameters for AutoMLConfig
# NOTE: DO NOT CHANGE THE experiment_timeout_minutes PARAMETER OR YOUR INSTANCE WILL TIME OUT.
# If you wish to run the experiment longer, you will need to run this notebook in your own
# Azure tenant, which will incur personal costs.
automl_config = AutoMLConfig(
    experiment_timeout_minutes=10,
    task='classification',
    primary_metric='accuracy',
    training_data=training_dataset,
    label_column_name='y',
    n_cross_validations=5,
    iterations=10,
    max_concurrent_iterations=8,
    compute_target=cpu_cluster)
automl_run = exp.submit(config=automl_config,tags=tag, show_output = True)
best_automl_run, best_model = automl_run.get_output()
#joblib.dump(best_automl_run, 'outputs/automl_bankmarketing_model.pkl')
# Get the metrics of the bestselected run
best_run_metrics = best_automl_run.get_metrics()
print(best_model._final_estimator)
# Show the Accuracy of that run
print('Best accuracy: {}'.format(best_run_metrics['accuracy']))