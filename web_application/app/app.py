import streamlit as st
from PIL import Image
import cv2
# from call_api import api_op
from aws_app import evaluate
import base64
from io import BytesIO
import numpy as np

RESIZE_AS=(320,240)

st.set_page_config(page_title="Stair_pose_estimation", page_icon="🤖")

st.title("Stair_pose")

chosen_option = st.sidebar.selectbox("Select Option", {"I want to upload an image", "Try out with inbuilt test images"})
st.write('chosen option : ',chosen_option)

if chosen_option == "Try out with inbuilt test images":
    chosen_image = st.sidebar.selectbox("Select Image", {"1001.png", "1002.png", "1477.png", "995.png", "front.png", "right.png", "left.png"})
    st.write('chosen image : ', chosen_image)

def main():
    if chosen_option == "I want to upload an image":
        st.write("I want to upload image")
        upload_image()
    if chosen_option == "Try out with inbuilt test images":
        try_using_inbuilt_image()
    # if chosen_option == "I want to upload a video":
    #     st.write("using uploaded video")
        # upload_video()

def upload_image():
    uploaded_file = st.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None: 
        uploaded_image_pil = Image.open(uploaded_file)
        uploaded_image_cv2 = np.array(uploaded_image_pil)
        st.image(uploaded_image_cv2, caption='original (uploaded) image')
        # buffered = BytesIO()
        # uploaded_image.save(buffered, format="PNG")
        # encoded_string = base64.b64encode(buffered.getvalue())
        stair_mask, pose = evaluate(uploaded_image_cv2)
        st.image(stair_mask, caption="mask of uploaded_image")
        st.write("pose is ", pose)
    #     resized_image=uploaded_image.resize((RESIZE_AS[0],RESIZE_AS[1]))
#         resized_image.save('ip_encode.png')
#         imghh=cv2.imread('ip_encode.png')
#         st.image(imghh)


# buffered = BytesIO()
# image.save(buffered, format="PNG")
# img_str = base64.b64encode(buffered.getvalue())
       
def try_using_inbuilt_image():
    path = "./" + chosen_image
    uploaded_image_bgr = cv2.imread(path)
    uploaded_image_rgb = cv2.cvtColor(uploaded_image_bgr, cv2.COLOR_BGR2RGB)  #since cv2 works with bgr, st.image works with rgb, PIL works with rgb
    st.image(uploaded_image_rgb, caption='original (uploaded) image')
    # resized_image = cv2.resize(uploaded_image_rgb, (RESIZE_AS[0],RESIZE_AS[1]), interpolation=cv2.INTER_NEAREST)
    # st.image(resized_image, caption="uploaded_image resized to (320,240)")
    # cv2.imwrite('ip_encode.png', uploaded_image_rgb)
    # with open("ip_encode.png", "rb") as image_file:
    #     encoded_string = base64.b64encode(image_file.read())
    stair_mask, pose = evaluate(uploaded_image_rgb)

    st.image(stair_mask, caption="mask of uploaded_image")
    st.write("pose is ", pose)

# def return_api_op(image):
#     decoded_mask, decoded_plot1, decoded_plot2, pose_dict = api_op(image)
#     decoded_mask_ = base64.b64decode(decoded_mask.encode('utf-8'))
#     with open('decoded_mask.png', 'wb') as f:
#         f.write(decoded_mask_)
#     decoded_plot1_ = base64.b64decode(decoded_plot1.encode('utf-8'))
#     with open('decoded_plot1.png', 'wb') as f:
#         f.write(decoded_plot1_)
#     decoded_plot2_ = base64.b64decode(decoded_plot2.encode('utf-8'))
#     with open('decoded_plot2.png', 'wb') as f:
#         f.write(decoded_plot2_)

#     final_mask = cv2.imread('decoded_mask.png')
#     final_plot1 = cv2.imread('decoded_plot1.png')
#     final_plot2 = cv2.imread('decoded_plot2.png')
    
#     st.image(final_mask, caption="mask of uploaded_image")
#     st.image(final_plot1, caption="vertical vanishing point")
#     st.image(final_plot2, caption="horizontal vanishing point")
#     st.write("""
#     Pose Calculation:
#     """)
#     st.write('Area of Staircase region is ', pose_dict["area"])
#     st.write('horiziontal_vanishing_point of staircase is ', pose_dict["horiziontal_vanishing_point"])
#     st.write('midpoint of last step of staircase is ', pose_dict["midpoint_last_step"])
#     st.write('slope of staircase is ', pose_dict["slope"])
#     st.write('vertical_vanishing_point of staircase is ', pose_dict["vertical_vanishing_point"])

#     # filename = 'ip_decode.png'  # I assume you have a way of picking unique filenames
#     # with open(filename, 'wb') as f:
#     #     f.write(decoded_mask)
    
#     # st.image(final_mask, caption="finalll maskkkk")

#     # final_mask=resnet_output(resized_image)

# def upload_video():
#     st.write("to be done")
    # uploaded_file = st.file_uploader("Upload file", type=".mp4")
    # take_ip_vid(uploaded_file)


if __name__ == "__main__":
    main()