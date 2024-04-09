import cv2
import numpy as np


def sliding_window(img1, img2, imgwidth, patch_size, istep=10):  # , jstep=1, scale=1.0):
    Ni, Nj = (int(s) for s in patch_size)
    for i in range(0, round(img1.shape[0] * 0.6) - Ni + 1, istep):
        patch = (img1[i:i + Ni, 0:imgwidth], img2[i:i + Ni, 0:imgwidth])
        yield (i, 0), patch


def Predict(patches):
    labels = np.zeros(len(patches))
    index = 0
    for Amplitude, theta in patches:
        mask = (Amplitude > 25).astype(np.float32)
        h, b = np.histogram(theta[mask.astype(np.bool)], bins=range(0, 80, 5))
        low, high = b[h.argmax()], b[h.argmax() + 1]
        newmask = ((Amplitude > 25) * (theta <= high) * (theta >= low)).astype(np.float32)
        value = ((Amplitude * newmask) > 0).sum()

        if value > 1500:
            labels[index] = 1
        index += 1
        # print(h)
        # print(low, high)
        # print(value)
        # cv2.imshow("newAmplitude", Amplitude * newmask)
        # cv2.waitKey(0)
    return labels


def preprocessing(image):
    """
Take the blue channel of the original image and filter it smoothly
    """
    kernel1 = np.ones((3, 3), np.uint8)
    kernel2 = np.ones((5, 5), np.uint8)
    gray = image[:, :, 0]
    gray = cv2.medianBlur(gray, 5)
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel1, iterations=4)
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel2, iterations=3)
    return gray


def getGD(canny):
    sobelx = cv2.Sobel(canny, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(canny, cv2.CV_32F, 0, 1, ksize=3)
    theta = np.arctan(np.abs(sobely / (sobelx + 1e-10))) * 180 / np.pi
    Amplitude = np.sqrt(sobelx ** 2 + sobely ** 2)
    mask = (Amplitude > 30).astype(np.float32)
    Amplitude = Amplitude * mask
    return Amplitude, theta


def getlocation(indices, labels, Ni, Nj):
    zc = indices[labels == 1]
    if len(zc) == 0:
        return 0, None
    else:
        xmin = int(min(zc[:, 1]))
        ymin = int(min(zc[:, 0]))
        xmax = int(xmin + Nj)
        ymax = int(max(zc[:, 0]) + Ni)
        return 1, ((xmin, ymin), (xmax, ymax))


def ZebraDetection(image):
    imgwidth = image.shape[1]
    Ni, Nj = (70, imgwidth)
    gray = preprocessing(image)
    canny = cv2.Canny(gray, 60, 360, apertureSize=3)
    Amplitude, theta = getGD(canny)
    indices, patches = zip(
        *sliding_window(Amplitude, theta, imgwidth, patch_size=(Ni, Nj)))
    labels = Predict(patches)
    indices = np.array(indices)
    ret, location = getlocation(indices, labels, Ni, Nj)
    return ret, location

if __name__=="__main__":
    img = cv2.imread('img.png')
    ret,location=ZebraDetection(img)
    print(ret)
    print(location)
    x1,x2,x3,x4=location
    print(x1)
    print(x2)