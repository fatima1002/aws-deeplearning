

# TODO: Install any packages that you might need
# For instance, you will need the smdebug package
!pip install smdebug

# TODO: Import any packages that you might need
# For instance you will need Boto3 and Sagemaker
import os
import sagemaker
import boto3
from sagemaker.pytorch import PyTorch
from sagemaker import get_execution_role
# from io import BytesIO
# from zipfile import *
from sagemaker.tuner import (
    IntegerParameter,
    CategoricalParameter,
    ContinuousParameter,
    HyperparameterTuner,
)
from sagemaker.debugger import (
    Rule,
    DebuggerHookConfig,
    rule_configs,
)

#TODO: Fetch and upload the data to AWS S3

# Command to download and unzip data
# !wget https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip

# --2022-01-06 19:50:15--  https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip
# Resolving s3-us-west-1.amazonaws.com (s3-us-west-1.amazonaws.com)... 3.5.163.157
# Connecting to s3-us-west-1.amazonaws.com (s3-us-west-1.amazonaws.com)|3.5.163.157|:443... connected.
# HTTP request sent, awaiting response... 200 OK
# Length: 1132023110 (1.1G) [application/zip]
# Saving to: ‘dogImages.zip’

# dogImages.zip       100%[===================>]   1.05G  38.8MB/s    in 43s     

# 2022-01-06 19:51:01 (24.9 MB/s) - ‘dogImages.zip’ saved [1132023110/1132023110]

# !unzip dogImages.zip
# Archive:  dogImages.zip
#    creating: dogImages/
#    creating: dogImages/test/
#    creating: dogImages/train/
#    creating: dogImages/valid/
#    creating: dogImages/test/001.Affenpinscher/
#   inflating: dogImages/test/001.Affenpinscher/Affenpinscher_00003.jpg  
#   inflating: dogImages/test/001.Affenpinscher/Affenpinscher_00023.jpg  
#   inflating: dogImages/test/001.Affenpinscher/Affenpinscher_00036.jpg  
#   inflating: dogImages/test/001.Affenpinscher/Affenpinscher_00047.jpg  
#   inflating: dogImages/test/001.Affenpinscher/Affenpinscher_00048.jpg  

# !aws s3 cp dogImages 's3://lesson4project/imageproject/' --recursive

# TODO: Set up debugging and profiling rules and hooks
from sagemaker.debugger import (
    Rule,
    DebuggerHookConfig,
    rule_configs,
)


hook_config = DebuggerHookConfig(
    hook_parameters={"train.save_interval": "100",
     "eval.save_interval": "10"}
)

# profiler, already imported Rule and rule_configs
from sagemaker.debugger import ( 
    ProfilerRule, 
)

from sagemaker.debugger import ProfilerConfig, FrameworkProfile
profiler_config = ProfilerConfig(
    system_monitor_interval_millis=500, 
    framework_profile_params=FrameworkProfile(num_steps=10)
)


# rewrite this
rules = [
    # profiler rules
    ProfilerRule.sagemaker(rule_configs.LowGPUUtilization()),
    ProfilerRule.sagemaker(rule_configs.ProfilerReport()),
    # debugger rules
    Rule.sagemaker(rule_configs.loss_not_decreasing()),
    Rule.sagemaker(rule_configs.vanishing_gradient()),
    Rule.sagemaker(rule_configs.overfit()),
    Rule.sagemaker(rule_configs.overtraining()),
    Rule.sagemaker(rule_configs.poor_weight_initialization()),
]

#TODO: Declare your HP ranges, metrics etc.
hyperparameter_ranges = {
    "lr": ContinuousParameter(0.001, 0.1),
    "batch-size": CategoricalParameter([64, 128, 256, 512]),
    "epochs": IntegerParameter(2, 4)
}

objective_metric_name = "average test loss"
objective_type = "Minimize"
metric_definitions = [{"Name": "average test loss", 
                       "Regex": "average test loss: ([0-9\\.]+)"}]


#TODO: Create estimators for your HPs
# bucket = 'lesson4project'
# prefix = "imageproject"
role = sagemaker.get_execution_role()

estimator = PyTorch(
    entry_point="train_model.py", #training file
    role=role, #execution role
    py_version='py36',
    framework_version="1.8",
    instance_count=1,
    instance_type="ml.m5.large",
    # adding this bc error
    debugger_hook_config = hook_config,
        ## Debugger parameters
    rules=rules,
    # profiler
    profiler_config=profiler_config,
    
)

tuner = HyperparameterTuner(
    estimator,
    objective_metric_name,
    hyperparameter_ranges,
    metric_definitions,
    max_jobs=4,
    max_parallel_jobs=2,
    objective_type=objective_type,
)

os.environ['SM_MODEL_DIR']='s3://lesson4project/imageproject/model/'
os.environ['SM_OUTPUT_DATA_DIR']='s3://lesson4project/imageproject/output/'
# TODO: Fit your HP Tuner :  # TODO: Remember to include your data channels
tuner.fit({"training": 's3://lesson4project/imageproject/'})




