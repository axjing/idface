import os
import time
from tqdm import tqdm
from PIL import Image
from models.facenet_api import Facenet
from utils.common import get_file_path, show_two_image
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
    WAIT=0.5
    start_i=time.time()
    model = Facenet()
    paths = get_file_path(face_dir, suffix="jpg")
    people_id = ["", float("inf")]
    print_time=0
    for i in tqdm(range(len(paths)), desc='Processing'):
        face_file_i = paths[i]
        image_1 = Image.open(input_im)
        image_2 = Image.open(face_file_i)
        similarity,inp_im,com_im = model.detect_image(image_1, image_2)
        # start_=time.time()
        show_two_image(inp_im,com_im,similarity[0],WAIT)
        # spend_=time.time()-start_
        # print_time+=spend_
        db_name = g_nm(face_file_i)[0]
        # print("{}\t---->\t{} | \tDistance: {} ".format(g_nm(input_im)[0], db_name, similarity))
        if similarity[0] < people_id[-1]:
            people_id[0] = db_name
            people_id[-1] = similarity[0]
        # time.sleep(WAIT)
        
    spend=time.time()-start_i-len(paths)*WAIT-print_time
    print("...\n\n")
    msg="Real ID:{} | Recognized ID:{} | Distance:{:.4f}|".format(g_nm(input_im)[0],people_id[0],people_id[-1])
    print("="*len(msg))
    print(msg)
    print("$ Spend Time: {:.3f}s".format(spend))
    print("="*len(msg))
    return people_id


if __name__ == "__main__":
    img = r"../data/刘亦菲.jpg"
    face_db = r"../data/face_db"
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
