import boto3
from io import BytesIO
from matplotlib import pyplot as plt
from matplotlib import image as mp_img

boto3 = boto3.Session()

def read_image_from_s3(bucket_name, image_name):

    # 画像情報を取得
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(name=bucket_name)
    object = bucket.Object(image_name)

    # 画像を取得して表示
    file_name = object.key
    file_stream = BytesIO()
    object.download_fileobj(file_stream)
    img = mp_img.imread(file_stream, format="jpeg")
    plt.imshow(img)
    plt.show()

    # 画像解析結果を取得
    client = boto3.client('rekognition', region_name="ap-northeast-1")
    response = client.detect_labels(Image={'S3Object': {'Bucket': bucket_name, 'Name': image_name}}, MaxLabels=10)

    print('Detected labels for ' + image_name)

    # ラベル情報を返す
    full_labels = response['Labels']
    return file_name, full_labels
