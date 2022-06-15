import os
from PIL import Image
from models.facenet_api import Facenet


def inference_face(input_im, face_im):
    model = Facenet()
    image_1 = Image.open(input_im)
    image_2 = Image.open(face_im)
    similarity = model.detect_image(image_1, image_2)
    print(similarity)
    return

def get_file_path(dir,suffix="jpg"):

    path_list=[]
    for f in os.listdir(dir):
        if f.split(".")[-1]==suffix:
            path_list.append(os.path.join(dir,f))
    return path_list

def inference_facedb(input_im,face_dir):
    """

    :param input_im:待识别人脸
    :param face_dir:人脸数据库文件夹
    :return:
    """
    model = Facenet()
    paths= get_file_path(face_dir,suffix="jpg")
    for i in range(len(paths)):
        face_file_i=paths[i]
        image_1 = Image.open(input_im)
        image_2 = Image.open(face_file_i)
        similarity = model.detect_image(image_1, image_2)
        print(similarity)

    return 0

if __name__ == "__main__":
    img=r"C:\Users\axjing\Pictures\xiaohai0.jpg"
    face=r"C:\Users\axjing\Pictures\xiaohai1.jpg"
    face_db=r"C:\Users\axjing\Pictures\face_db"
    # inference_face(img,face)
    inference_facedb(img,face_dir=face_db)
    # model = Facenet()
    # while True:
    #     image_1 = input('Input image_1 filename:')
    #     try:
    #         image_1 = Image.open(image_1)
    #     except:
    #         print('Image_1 Open Error! Try again!')
    #         continue
    #
    #     image_2 = input('Input image_2 filename:')
    #     try:
    #         image_2 = Image.open(image_2)
    #     except:
    #         print('Image_2 Open Error! Try again!')
    #         continue
    #
    #     probability = model.detect_image(image_1, image_2)
    #     print(probability)
