import time
import streamlit as st
from PIL import Image
import numpy as np
from models.facenet_api import Facenet
from utils.common import get_file_path, show_two_image
from utils.common import get_filename_in_path as g_nm

st.set_page_config(
    page_title="Anders FaceID System v1",
    page_icon=":pencil:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://mp.weixin.qq.com/mp/homepage?__biz=MzI0OTIwMTY3OA==&hid=2&sn=9e8edebde0f1249563eca4c1864dbd0d&scene=18',
        'Report a bug': "https://mp.weixin.qq.com/mp/homepage?__biz=MzI0OTIwMTY3OA==&hid=2&sn=9e8edebde0f1249563eca4c1864dbd0d&scene=18",
        'About': "https://mp.weixin.qq.com/mp/homepage?__biz=MzI0OTIwMTY3OA==&hid=2&sn=9e8edebde0f1249563eca4c1864dbd0d&scene=18"
    }
)
st.title('Anders FaceID System')
st.write("小应用：通用人脸识别")
st.text("手机访问请把该链接复制到手机浏览器使用")


def image_input():
    WAIT=0
    # if st.sidebar.checkbox('Upload'):
    #     content_file = st.sidebar.file_uploader("Choose a Content  Image", type=["png", "jpg"])
    # else:
    #     content_file = st.sidebar.file_uploader("Choose a Content  Image", type=["png", "jpg"])
    content_file = st.sidebar.file_uploader("Choose a Content  Image", accept_multiple_files=False,type=["png", "jpg"])
    if content_file is not None:
        # To read file as bytes:
        image_1 = Image.open(content_file)
        st.markdown("## 申请进入")
        st.image(image_1)
        start = time.time()
        model = Facenet()
        face_dir = r"../data/face_db"
        paths = get_file_path(face_dir, suffix="jpg")
        people_id = ["", float("inf"),0]

        # # 添加占位符
        # placeholder = st.empty()
        # # 创建进度条
        # bar = st.progress(0)
        for i in range(len(paths)):
            face_file_i = paths[i]
            image_2 = Image.open(face_file_i)
            similarity, inp_im, com_im = model.detect_image(image_1, image_2)
            # st.image(com_im)
            db_name = g_nm(face_file_i)[0]
            # print("{}\t---->\t{} | \tDistance: {} ".format(g_nm(input_im)[0], db_name, similarity))
            if similarity[0] < people_id[1]:
                people_id[0] = db_name
                people_id[1] = similarity[0]
                people_id[2]=com_im

            # # 不断更新占位符的内容
            # placeholder.text(f"Processing {i + 1}")
            # # 不断更新进度条
            # bar.progress(i + 1)

        spend = time.time() - start - len(paths) * WAIT
        st.markdown("## 识别结果")

        msg = "Recognized ID:{} | Distance:{:.4f}|\n".format(people_id[0], people_id[1])

        if people_id[1]>1.13:
            st.warning("This person doesn't exist in the face database\n$ Spend Time: {:.3f}s".format(spend))
        else:
            st.image(people_id[2])
            msg+="\n$ Spend Time: {:.3f}s".format(spend)
            st.success(msg)

        # st.image(content)
        # hist = content.histogram()
        # st.line_chart(hist)
        # reader = easyocr.Reader(['ch_sim', 'en'], gpu=False,
        #                         download_enabled=True)  # this needs to run only once to load the model into memory
        # with st.spinner('Wait Processing...'):
        #     result = reader.readtext(np.array(content), detail=0)
        #
        # s = ""
        # if len(result) == 0:
        #     st.warning('图片中没有文字内容!')
        # else:
        #     for i in result:
        #         st.write(i)
        #         s += i + '\n'
        #     st.success('OCR is a success message!')
        #     st.info(s)
        # st.write("Time:{:.2f}".format(time.time() - start))
    else:
        st.warning("Upload an Image OR Untick the Upload Button")
        st.stop()
if __name__=="__main__":
    image_input()