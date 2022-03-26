import cv2

calib_image_list = './calib_list.txt'
calib_batch_size = 10

def calib_input(iter):
  images = []
  line = open(calib_image_list).readlines()
  for index in range(0, calib_batch_size):
    curline = line[iter * calib_batch_size + index]
    calib_image_name = curline.strip()

    # open image as BGR
    image = cv2.imread('img/'+str(calib_image_name))
    image = cv2.resize(image, (150, 150))

    # change to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.reshape((150, 150, 3))

    # normalize
    image = image/255.0

    images.append(image)
  return {"placeholder": images}


