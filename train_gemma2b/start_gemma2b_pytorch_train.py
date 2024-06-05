import datetime
from sagemaker.pytorch import PyTorch
import boto3
import yaml


def get_config(config: str):
    # Reads a YAML configuration file.
    with open(f"{config}.yaml", "r") as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


config = get_config("../config")
iam = boto3.client('iam', region_name='us-east-1')
role = iam.get_role(RoleName='sohyung_tryout')['Role']['Arn']

model_name = 'google/gemma-2b'
tokenizer_name = model_name

ct = datetime.datetime.now()
current_time = str(ct.now()).replace(":", "-").replace(" ", "-")[:19]
training_job_name = f'finetune-{model_name}-{current_time}'

hyperparameters={
    'output_dir': '/opt/ml/checkpoints',
}

huggingface_estimator = PyTorch(
    entry_point='start_gemma2b_pytorch_train.py',
    source_dir='./scripts',
    instance_type='ml.p3.2xlarge',
    instance_count=1,
    role=role,
    framework_version='2.2', # Gemma2b requires > Pytorch 2.1.1
    py_version='py310',
    max_run=60 * 60 * 2,  # expected max run in seconds
    checkpoint_s3_uri='s3://sagemaker-us-east-1-211125645004/checkpoints/',
    environment={
        'HF_TOKEN': config['token']  # Set the HF_TOKEN environment variable here
    }
)

# Stingning ultracht dataset - only train dataset
print(f"Starting job with name: {training_job_name}")
s3_train_path = f's3://sagemaker-us-east-1-211125645004/samples/datasets/stingning/ultrachat/train/'

# Launch the training job
huggingface_estimator.fit({'train': s3_train_path}, job_name=training_job_name)