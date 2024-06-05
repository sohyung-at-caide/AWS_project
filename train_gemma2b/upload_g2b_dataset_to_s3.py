from datasets import load_dataset
import aiobotocore.session
import s3fs


def main():
    # Define the name of the dataset to be loaded
    dataset_name = "stingning/ultrachat"
    train_dataset = load_dataset(dataset_name, split="train[:10000]")

    # Create an asynchronous session with my AWS profile
    s3_session = aiobotocore.session.AioSession(profile="sagemaker-user")
    storage_options = {"session": s3_session}
    fs = s3fs.S3FileSystem(**storage_options)

    # Upload to S3
    bucket = "sagemaker-us-east-1-211125645004"
    s3_prefix = f'samples/datasets/{dataset_name}'
    training_input_path = f's3://{bucket}/{s3_prefix}/train'
    train_dataset.save_to_disk(training_input_path, fs=fs)

    print(f'Uploaded training data to {training_input_path}')


if __name__ == '__main__':
    main()
