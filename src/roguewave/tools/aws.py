import boto3

def aws_byte_range(bucket, key,byterange):
    obj = boto3.resource('s3').Object(bucket, key)
    stream = obj.get(Range=f'bytes={byterange[0]}-{byterange[1]}')['Body']
    return stream.read()
