import json
import logging
import mimetypes
import os

from minio import S3Error, Minio

from config import minio_config

logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.DEBUG)
logger = logging.getLogger('minio_client')

BASEDIR = os.path.abspath(os.path.dirname(__file__))


def _set_bucket_public_policy(minio_client, bucket):
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"AWS": "*"},
                "Action": ["s3:GetBucketLocation", "s3:ListBucket"],
                "Resource": "arn:aws:s3:::my-bucket",
            },
            {
                "Effect": "Allow",
                "Principal": {"AWS": "*"},
                "Action": "s3:GetObject",
                "Resource": "arn:aws:s3:::my-bucket/*",
            },
        ],
    }
    statements = policy.get('Statement')
    for statement in statements:
        statement['Resource'] = statement.get('Resource').replace('my-bucket', bucket)
    policy['Statement'] = statements
    minio_client.set_bucket_policy(bucket, json.dumps(policy))


def _guess_content_type(file_path):
    (mime_type, _) = mimetypes.guess_type(file_path)
    if mime_type:
        return mime_type
    else:
        return 'application/octet-stream'


def _upload(minio_client, bucket_name, file_name, file_path, content_type=None):
    logger.info(minio_client, bucket_name, file_name, file_path, content_type)

    bucket = minio_client.bucket_exists(str(bucket_name))
    file_path += '/' + file_name

    if not bucket:
        logger.info(f"Bucket {bucket_name} not found. Creating a new one.")
        minio_client.make_bucket(bucket_name)
        _set_bucket_public_policy(minio_client, bucket_name)
    else:
        logger.info(f"Bucket {bucket_name} found")

    if not content_type:
        content_type = _guess_content_type(file_path)
        logger.info(f"Content type: {content_type}")

    try:
        logger.info(f"Saving file {file_name} ({content_type}) to bucket {bucket_name} from {file_path}.")
        minio_client.fput_object(
            bucket_name=bucket_name,
            object_name=file_name,
            file_path=file_path,
            content_type=content_type
        )
        logger.info(f"{file_name} is successfully uploaded!")
        return file_name
    except S3Error as exception:
        logger.error(f"Error occurred: {exception}")
        return None


def _setup_minio():
    minio_setup = Minio(
        minio_config.get('SERVER_HOST'),
        minio_config.get('ACCESS_KEY'),
        minio_config.get('SECRET_KEY'),
        secure=minio_config.get('SECURE')
    )
    return minio_setup


def upload_file(bucket_name, file_name, file_path):
    minio_client = _setup_minio()
    try:
        return _upload(minio_client, bucket_name, file_name, file_path)
    except S3Error as exception:
        logger.error(f"Error occurred: {exception}")
        return None
