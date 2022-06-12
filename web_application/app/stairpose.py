import cv2
import numpy as np
from itertools import permutations
from itertools import combinations
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import io
import base64
import streamlit as st


def position_mask(mask, m_orig, m, n_orig, n, ymin, ymax, xmin, xmax):
    # places mask to orig image
    unpadded_mask = mask[(m_orig-m)//2+1:(m_orig-m)//2 +
                         1+m, (n_orig-n)//2+1:(n_orig-n)//2+1+n]
    black_image = np.zeros((240, 320), dtype=np.uint8)
    black_image[ymin:ymax, xmin:xmax] = unpadded_mask[:, :]
    positioned_mask_image = black_image
    return positioned_mask_image


def hough_transform(image):
    """
    Determine and cut the region of interest in the input image.
    Parameters:
            image: The output of a Canny transform.
    """
    rho = 1  # Distance resolution of the accumulator in pixels.
    theta = np.pi/180  # Angle resolution of the accumulator in radians.
    # Only lines that are greater than threshold will be returned.
    threshold = 87
    minLineLength = 10  # Line segments shorter than that are rejected.
    maxLineGap = 350  # Maximum allowed gap between points on the same line to link them
    return cv2.HoughLinesP(image, rho=rho, theta=theta, threshold=threshold,
                           minLineLength=minLineLength, maxLineGap=maxLineGap)


def polyarea(x, y):
    # polyarea([0,0,4,4],[0,4,4,0])
    return 0.5*np.abs(np.dot(x, np.roll(y, 1))-np.dot(y, np.roll(x, 1)))
# PolygonArea([[0,0],[0,4],[4,4],[4,0]])


def get_intersection(i):
    x1 = i[0][0]
    y1 = i[0][1]
    x2 = i[0][2]
    y2 = i[0][3]
    x3 = i[1][0]
    y3 = i[1][1]
    x4 = i[1][2]
    y4 = i[1][3]
    if (x1-x2) == 0 or (x3-x4) == 0:  # one or two lines detected is point
        return(0, 0)
    else:
        b = (y1-y2)/(x1-x2)
        a = (y3-y4)/(x3-x4)
        a1 = (y3-a*x3)
        b1 = (y1-b*x1)
        if (b-a) == 0:  # slope diff =0 means lines are
            return (0, 0)
        else:
            X = (a1-b1)/(b-a)
            Y = b*(X-x1)+y1
            return (X, Y)


def process_mask(mask, ymin, ymax, xmin, xmax, m_orig, n_orig, m, n, uploaded_image_shape):
    # st.image(mask, caption= "mask in func_process_mask")
    stair_mask = position_mask(
        mask, m_orig, m, n_orig, n, ymin, ymax, xmin, xmax)
    stair_mask = cv2.resize(
        stair_mask, (uploaded_image_shape[1], uploaded_image_shape[0]), interpolation=cv2.INTER_AREA)
    # stair_mask = np.zeros((240, 320), np.uint8)
    # stair_mask[ymin:ymax, xmin:xmax] = mask[ymin:ymax, xmin:xmax]
    # st.image(mask, caption= "stair_mask in func_process_mask")
    # mask.dtype = int64 and stair_mask.dtype = uint8 ,and  cv2.connectedComponentsWithStats was giving error with int64 dtype :
    # """ -215:Assertion failed) iDepth == CV_8U || iDepth == CV_8S in function 'cv::connectedComponents_sub1 """
    # so above code is written to convert into uint8 dtype
    img = np.stack((stair_mask, stair_mask, stair_mask), axis=-1)
    img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT)

    # find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
        img[:, :, 0], connectivity=8)  # connectivity =4(erodes more) or 8(erodes less)
    # connectedComponentswithStats yields every seperated component with information on each of them, such as size
    # the following part is just taking out the background w[[275, 175, 559, 200]],which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    # minimum size of particles we want to keep (number of pixels)
    # here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
    min_size = 20
    img2 = np.zeros((img.shape), dtype=np.uint8)
    img3 = np.zeros((img.shape), dtype=np.uint8)
    # img4 = np.zeros((img.shape),dtype=np.uint8)
    # for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255

    lines = hough_transform(img2[:, :, 0])
    # try:
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0][0], line[0][1], line[0][2], line[0][3]
            cv2.line(img3, (x1, y1), (x2, y2), (255, 255, 255), 2)

            # THESE X1,Y1 & X2,Y2 VALUES WILL BE USED USED TO CALCULATE HORIZONTAL POINTS(AS EQN CAN BE FOUND FROM HERE N THEN NC2 POINT INTERSECTION ND USKA MEAN/MEDIAN/MODE)
            # cv2.circle(img4,(x1,y1),3,[0,255,0],-1)
            # cv2.circle(img4,(x2,y2),3,[0,255,0],-1)

    operatedImage = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    dest = cv2.cornerHarris(operatedImage, 5, 31, 0.2)
    dest = cv2.dilate(dest, None)
    result = np.where(dest > 0.02 * dest.max())
    list1 = []
    list2 = []
    boolean = True
    for num in range(result[0].shape[0]):
        while boolean:
            list1.append(result[0][num])
            list2.append(result[1][num])
            boolean = False
        # TRY REMOVING ABSOLUTE
        if np.abs(result[0][num]-list1[-1]) > 5 or np.abs(result[1][num]-list2[-1]) > 5:
            list1.append(result[0][num])
            list2.append(result[1][num])
    for num in range(len(list1)):
        cv2.circle(img3, (list2[num], list1[num]),
                   3, [0, 255, 0], -1)  # BGR FORMAT
    # img_concate_Hori=np.concatenate((img,img3),axis=1)
    # cv2.imshow('1',img)
    # cv2.imshow('2',img3)
    # calculation of area of stair
    stair_area = polyarea(list1, list2)
    label_html = ' (1) stair case area is ' + str(stair_area) + '\n'
    pose = {"area": stair_area}
    #---------------------------------------------------------------------------print('stair case area is {}'.format(stair_area))
    # Seperating left and right corners of stair
    X = np.array(list(zip(list2, list1)))
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(X)
    res = list(zip(*X[kmeans.labels_ == 0]))
    res1 = list(zip(*X[kmeans.labels_ == 1]))
    for num in range(len(res[0])):
        cv2.circle(img3, (res[0][num], res[1][num]), 3, [255, 0, 0], -1)
    for num in range(len(res1[0])):
        cv2.circle(img3, (res1[0][num], res1[1][num]), 3, [0, 0, 255], -1)

    # cv2.circle(img3,( int((res[0][0]+res1[0][0])/2) , int((res[1][0]+res1[1][0])/2) ),3,[0,255,255],-1)
    cv2.circle(img3, (int((res[0][-1]+res1[0][-1])/2),
               int((res[1][-1]+res1[1][-1])/2)), 4, [0, 255, 255], -1)
    label_html = label_html + ' (2) Mid point of last step i.e. closer to ground is (' + str(
        int((res[0][-1]+res1[0][-1])/2)) + ',' + str(int((res[1][-1]+res1[1][-1])/2)) + ')' + '\n'
    pose["midpoint_last_step"] = [
        int((res[0][-1]+res1[0][-1])/2), int((res[1][-1]+res1[1][-1])/2)]
    # img_concate_3=np.concatenate((img_concate_Hori,img3),axis=1)
    # cv2.imshow('3',img3)
    # plt.show()

    fig, ax = plt.subplots(figsize=(6, 5))
    plt.gca().invert_yaxis()  # to shift origin to upperleft corner
    x = np.array(res[0]).reshape(-1, 1)
    y = np.array(res[1]).reshape(-1, 1)
    x1 = np.array(res1[0]).reshape(-1, 1)
    y1 = np.array(res1[1]).reshape(-1, 1)
    reg = LinearRegression()
    reg.fit(x, y)
    # ax.scatter(x,y,c='blue')
    reg1 = LinearRegression()
    reg1.fit(x1, y1)
    # ax.scatter(x1,y1,c='red')
    c = np.squeeze(reg1.coef_)
    d = np.squeeze(reg1.intercept_)
    b = np.squeeze(reg.intercept_)
    a = np.squeeze(reg.coef_)

    # print("The linear model is: Y = {:.5} + {:.5}X".format(b, a))  # b a
    # print("The linear model is: Y = {:.5} + {:.5}X".format(d, c))  # d c

    # calculation of theta i.e angle between two lines
    slope = np.arctan(np.abs((a-c)/(1 + a*c)))
    slope = (slope*180)/np.pi
    #---------------------------------------------------------------------------print("The angle between these two lines is {}".format(slope))
    label_html = label_html + \
        ' (3) The angle between these two lines is ' + str(slope) + '\n'
    pose["slope"] = slope

    try:
        predictions = np.squeeze(reg.intercept_)+np.squeeze(reg.coef_) * x
        predictions1 = np.squeeze(reg1.intercept_)+np.squeeze(reg1.coef_) * x1
        plt.plot(x, predictions, c='green', linewidth=2)
        plt.plot(x1, predictions1, c='green', linewidth=2)

        X_intersection = (d-b)/(a-c)
        Y_intersection = a*X_intersection+b
        label_html = label_html + ' (4) vertical vanishing point is ' + \
            str(X_intersection) + ',' + str(Y_intersection) + '\n'
        pose["vertical_vanishing_point"] = [X_intersection, Y_intersection]
        plt.scatter(X_intersection, Y_intersection, marker='*', s=100)
        # my_stringIObytes = io.BytesIO()
        # plt.savefig(my_stringIObytes, format='jpg')
        # my_stringIObytes.seek(0)
        # plot_string1 = base64.b64encode(my_stringIObytes.read())
        #-------------------------------------------------------------------------print('vertical vanishing point is {}'.format((X_intersection,Y_intersection)))
        plt.show()
        st.write(fig)
    except:
        print('error while calculating vertical vanishing point')
        plot_string1 = "error"

    # calculate horizontal intersection point
    try:
        line_cords = []
        lines = hough_transform(img3[:, :, 0])
        if lines is not None:

            fig1 = plt.figure()
            plt.gca().invert_yaxis()  # to shift origin to upperleft corner
            for line in lines:
                # if random() > 0.2:
                x1, y1, x2, y2 = line[0][0], line[0][1], line[0][2], line[0][3]
                line_cords.append((x1, y1, x2, y2))
                x = [x1, x2]
                y = [y1, y2]
                plt.plot(x, y, c='green')

        count = 0
        x0 = 0
        y0 = 0
        comb_line_cords = list(combinations(line_cords, 2))
        for i in comb_line_cords:
            if get_intersection(i) == (0, 0):
                # count-=1
                # print("printed corresp. to (0,0)")
                pass
            else:
                x0 += get_intersection(i)[0]
                y0 += get_intersection(i)[1]
                count += 1
        # x_mean=x0/(count+2)
        # y_mean=y0/(count+2)
        x_mean = x0/(count+1)
        y_mean = y0/(count+1)

        label_html = label_html + \
            ' (5) horizontal vanishing point is ' + \
            str(x_mean) + ',' + str(y_mean)
        pose["horiziontal_vanishing_point"] = [x_mean, y_mean]
        plt.scatter(x_mean, y_mean, marker='*', s=100)
        # first
        plt.ylim(240, 0)
        plt.show()
        st.write(fig1)
        # -------------------------------------------------------------------------print('horizontal vanishing point is {}'.format((x_mean,y_mean)))
        # my_stringIObytes = io.BytesIO()
        # plt.savefig(my_stringIObytes, format='jpg')
        # my_stringIObytes.seek(0)
        # plot_string2 = base64.b64encode(my_stringIObytes.read())
    except:
        st.write('error while calculating horizontal vanishing point')
        # plot_string2 = "error"
    # ------------------------------------------------------------------------print('done2')
    # --------------------------------------------------------------------------print(label_html)
    return pose, stair_mask

    # except:
    #   print('problem with stair edge detection')
    #   label_html = 'No stair edge detected'
    #   return label_html
