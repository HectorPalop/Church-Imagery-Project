import requests
import urllib.request
import pandas as pd
import os
import json
import numpy as np
import cv2
import math


def concatenate_list_data(list):
    result= ''
    for element in list:
        result += str(element)
    return result

def getCoordinates(coordinates):
    URLarr = ["https://nominatim.openstreetmap.org/search.php?",
            "q=" + coordinates,
            "&polygon_geojson=1&format=json"]
    URL = concatenate_list_data(URLarr)
    return URL

def getURL(coordinates, maptype, polyline, key, base64_signature):
    zoom = "19"
    size1 = "600"
    size2 = "300"
    format = "png32"
    URLarr = ["https://maps.googleapis.com/maps/api/staticmap?",
            "center=" + coordinates,
            "&zoom=" + zoom,
            "&size=" + size1 + "x" + size2,
            "&format=" + format,
            "&maptype=" + maptype,
            "&style=feature:all|element:labels|visibility:off",
            "&sensor=false",
            "&key=" + key,
            "&signature=" + base64_signature,
            polyline]
    URL = concatenate_list_data(URLarr)
    return URL

def getPolyline(coord, iter, userPath, ident):
    os.chdir(userPath)
    URL = getCoordinates(coordinates = coord)
    url = 'http://www.reddit.com/r/all/top/.json'
    req = urllib.request.Request(URL)
    ##parsing response
    r = urllib.request.urlopen(req).read()
    cont = json.loads(r.decode('utf-8'))
    counter = 0
    ##parsing json
    polyline = np.asarray(cont[0]['geojson']['coordinates'])
    return polyline

def transformCoordinates(polyline):
    newPoly = []
    color = '0xff0000ff'
    weight = '1'
    fill = 'none'
    for lat, lon in polyline[0]:
        newStr = '|' + str(lon) + ',' + str (lat)
        newPoly.append(newStr)
    coords = concatenate_list_data(newPoly)
    polyString = '&path=color:'+color+'|weight:'+weight+'fillcolor:'+fill+coords
    return polyString


def getImagery(coord, mtype, iter, userPath, ident, poly, key, signature):
    if (mtype == 'satellite'):
        tb = 'sat'
    else:
        if (mtype == 'roadmap'):
            tb = 'rdm'
        else:
            if (mtype == 'polygon'):
                tb = 'pol'
                mtype = 'satellite'
            else:
                tb = 'otr'
    url = getURL(coordinates = coord, maptype = mtype, polyline = poly, key=key, base64_signature=signature)
    os.chdir(userPath)
    r = requests.get(url, allow_redirects=True)
    filename = tb + str(iter) + ".png"
    open(filename, 'wb').write(r.content)
    print ("Downloaded ", mtype ," sample #", iter, "URL: " , url, "\n")

def contrast_brightness(alpha, beta, image):
    new_image = image.copy()
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            for c in range(image.shape[2]):
                new_image[y,x,c] = np.clip(alpha*image[y,x,c] + beta, 0, 255)
    return new_image

def retrieve_local_path(local_adress, type, num):
    path = [local_adress,
            type,
            str(num),
            '.png']
    local_path = concatenate_list_data(path)
    return local_path

def hue_filter(img, mask_low, mask_high):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, mask_low, mask_high)
    mask_inv = cv2.bitwise_not(mask)
    new_img = cv2.bitwise_and(img, img, mask = mask_inv)
    return new_img

def binary_filter(img, threshold):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    retval, threshold = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    return threshold

def getMainContour(img, contours):
    cix = int (img.shape[1] / 2)
    ciy = int (img.shape[0] / 2)
    minDist = img.shape[0]
    minArea = 0
    totArea = img.shape[1] * img.shape[0]
    biggestContour = []
    mainContour = []
    for contour in contours:
        M = cv2.moments(contour)
        localArea = cv2.contourArea(contour)
        if ((localArea<1/3.5*totArea) and (localArea>1/256*totArea)):
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            dist = math.hypot(cx - cix, cy - ciy)
            if ((dist < img.shape[0]/4) and (minArea < localArea)):
                minArea = localArea
                mainContour = biggestContour
                biggestContour = contour
    return mainContour

def getMainContour2(img, contours):
    cix = int (img.shape[1] / 2)
    ciy = int (img.shape[0] / 2)
    minDist = img.shape[0]
    minArea = 0
    totArea = img.shape[1] * img.shape[0]
    biggestContour = []
    mainContour = []
    contour_flag = False
    for contour in contours:
        M = cv2.moments(contour)
        localArea = cv2.contourArea(contour)
        if ((M["m00"]!=0) and (localArea<1/2*totArea) and (localArea>1/256*totArea)):
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            dist = math.hypot(cx - cix, cy - ciy)
            if (dist < minDist):
                minDist = dist
                mainContour = contour
                mcx = cx
                mcy = cy
                contour_flag = True

    if (contour_flag == False):
        P1=[[1 * img.shape[1] / 4,   1 * img.shape[0] / 4]]
        P2=[[3 * img.shape[1] / 4,   1 * img.shape[0] / 4]]
        P3=[[3 * img.shape[1] / 4,   3 * img.shape[0] / 4]]
        P4=[[1 * img.shape[1] / 4,   3 * img.shape[0] / 4]]
        P5=P1
        mainContour = np.array([P1,  P2,  P3,  P4, P5], np.int32)

    return mainContour, contour_flag

def obscure_background(img, shape, color):
    ctrs = np.asarray(shape).astype(np.int)
    stencil = np.zeros(img.shape).astype(img.dtype)
    cv2.fillPoly(stencil, [ctrs], color)
    result = cv2.bitwise_and(img, stencil)
    return result

def extract_contour(impol):
    imcol = hue_filter(img = impol, mask_low = (0,0,0), mask_high = (254,254,255))
    imbin = binary_filter(img = imcol, threshold = 1)
    immed = cv2.medianBlur(imbin,5)
    contours,hierarchy = cv2.findContours(immed, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    mainContour = getMainContour(img = impol, contours = contours)
    return mainContour

def extract_rdm_contour(img):
    avbgr = img.mean(axis=0).mean(axis=0)
    if (avbgr[0] < 240):
        sat_val = 2.4
        threshval = 240
        bri_val = -300
    else:
        sat_val = 5.4
        threshval = 254
        bri_val = -1050
    imgcb = contrast_brightness(alpha=sat_val, beta=bri_val, image=img)
    gray = cv2.cvtColor(imgcb,cv2.COLOR_BGR2GRAY)
    retval, threshold = cv2.threshold(gray, threshval, 255, cv2.THRESH_BINARY_INV)
    contours,hierarchy = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    mCont, contour_flag = getMainContour2(img = img, contours = contours)
    return mCont, contour_flag

def calculate_reject_index(img):
    cix = int (img.shape[1] / 2)
    ciy = int (img.shape[0] / 2)
    rj = 2 * int(img[ciy, cix][2])-int(img[ciy, cix][0])-int(img[ciy, cix][1])
    return rj

###############################################################################################
#################################         INTERFACE       #####################################
coord_path = '[Your path to the .xlsx document with the church coordinates goes here]'
download_folder_path = '[Your path to the download imagery folder goes here]'
API_key = '[Your API key goes here]'
signature= '[Your digital signature goes here]'
###############################################################################################

#reading the excel file
df = pd.read_excel(coord_path)
#building the URL and downloading the file
no_poly = []
yes_poly = []
for i in range (df.shape[0]):
    latlong = pd.Series([str(df.at[i,'Latitude']),str(df.at[i,'Longitude'])]).str.cat(sep=',')
    polyline = getPolyline(userPath = download_folder_path, coord=latlong, iter=i, ident=str(df.at[i,'Balkerne_Reference']))
    if (len(polyline.shape) != 3):
        print('Sample # ',i,': Polyline does not exist! \n')
        no_poly.append(i)
    else:
        strPoly = transformCoordinates (polyline)
        getImagery(userPath = download_folder_path, coord=latlong, mtype='polygon', iter=i, ident=str(df.at[i,'Balkerne_Reference']), poly=strPoly, key=API_key)
        getImagery(userPath = download_folder_path, coord=latlong, mtype='satellite', iter=i, ident=str(df.at[i,'Balkerne_Reference']), poly="", key=API_key)
        yes_poly.append(i)

print ('Failed to obtain the polyline of', len(no_poly), 'samples:', no_poly)
print ('Succesful polyline in', len(yes_poly), 'samples:', yes_poly)

#######################################################################################################################################
invalid_contours=[]

for poly_sample in yes_poly:

    sat_path = retrieve_local_path(local_adress = download_folder_path, type = 'sat', num = poly_sample)
    pol_path = retrieve_local_path(local_adress = download_folder_path, type = 'pol', num = poly_sample)
    img_satellite = cv2.imread(sat_path)
    img_polygon = cv2.imread(pol_path)

    if(isinstance(img_polygon, (list, tuple, np.ndarray))):
        contour = extract_contour(impol = img_polygon)
    else:
        contour = []

    if (len(contour)!=0):
        imobs=obscure_background(img = img_satellite, shape = contour, color = [255, 255, 255])
        filename = "fil" + str(poly_sample) + ".png"
        cv2.imwrite(filename, imobs)
    else:
        print ('Sample', poly_sample, 'does not have a valid contour')
        invalid_contours.append(poly_sample)

    os.remove(sat_path)
    os.remove(pol_path)

print ('Invalid contour detected in', len(invalid_contours), 'samples:', invalid_contours)

#######################################################################################################################################

failed_samples = no_poly + invalid_contours
print('Samples with invalid polygons:', failed_samples, ', attempting to rescue.')

for fSample in failed_samples:

    latlong = pd.Series([str(df.at[fSample,'Latitude']),str(df.at[fSample,'Longitude'])]).str.cat(sep=',')
    getImagery(userPath = download_folder_path, coord=latlong, mtype='satellite', iter=fSample, ident=str(df.at[fSample,'Balkerne_Reference']), poly="", key=API_key)
    getImagery(userPath = download_folder_path, coord=latlong, mtype='roadmap', iter=fSample, ident=str(df.at[fSample,'Balkerne_Reference']), poly="", key=API_key)

#######################################################################################################################################


rejected_list = []

for fSample in failed_samples:
    rdm_path = retrieve_local_path(local_adress = download_folder_path, type = 'rdm', num = fSample)
    sat_path = retrieve_local_path(local_adress = download_folder_path, type = 'sat', num = fSample)

    imrdm = cv2.imread(rdm_path)
    imsat = cv2.imread(sat_path)
    rdm_contour, contour_flag = extract_rdm_contour(img = imrdm)
    imobs=obscure_background(img = imsat, shape = rdm_contour, color = [255, 255, 255])

    filename = "fil" + str(fSample) + ".png"
    cv2.imwrite(filename, imobs)

    reject_index = calculate_reject_index(img = imrdm)
    if ((reject_index < -10) or (contour_flag==False)):
        rejected_list.append(fSample);
        print ('Failed to rescue image:', fSample)
    else:
        print ('Rescued image:', fSample)

    os.remove(sat_path)
    os.remove(rdm_path)

print ('\n---------------------------------------\n')
print (len (rejected_list), 'images need manual revision:', rejected_list)
