import os
from PIL import Image
from models.facenet_api import Facenet
from utils.common import get_file_path
from utils.common import get_filename_in_path as g_nm


def inference_face(input_im, face_im):
    model = Facenet()
    image_1 = Image.open(input_im)
    image_2 = Image.open(face_im)
    similarity = model.detect_image(image_1, image_2)
    print(similarity)
    return


def inference_facedb(input_im, face_dir):
    """

    :param input_im:待识别人脸
    :param face_dir:人脸数据库文件夹
    :return:
    """
    model = Facenet()
    paths = get_file_path(face_dir, suffix="jpg")
    people_id = ["", 1000000]
    for i in range(len(paths)):
        face_file_i = paths[i]
        image_1 = Image.open(input_im)
        image_2 = Image.open(face_file_i)
        similarity = model.detect_image(image_1, image_2)
        db_name = g_nm(face_file_i)[0]
        print("The distance from {} to {} is {} ".format(g_nm(input_im)[0], db_name, similarity))
        if similarity < people_id[-1]:
            people_id[0] = db_name
            people_id[-1] = similarity

    print("="*50)
    print("Identity:{},Distance:{}".format(people_id[0],people_id[-1]))
    print("="*50)

    return people_id


if __name__ == "__main__":
    face = r"C:\Users\axjing\Pictures\xiaohai1.jpg"
    # img=r"C:\Users\axjing\Pictures\xiaohai0.jpg"

    img = r"C:\Users\admin\Pictures\liu.jpg"
    face_db = r"C:\Users\admin\Pictures\face_db"
    # inference_face(img,face)
    inference_facedb(img, face_dir=face_db)
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
