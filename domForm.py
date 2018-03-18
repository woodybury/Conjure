import cv2, time
import numpy as np
from occamy import Socket
from threading import Thread
from listening import recognition

socket = Socket("ws://dlevs.me:4000/socket")
socket.connect()

channel = socket.channel("room:lobby", {})
channel.on("connect", print ('Im in'))
channel.on("new_msg", lambda msg, x: print("> {}".format(msg["body"])))

channel.join()

def transform( image ):
    img = cv2.imread(image, 0)

    # resize img for transform
    img = cv2.resize(img, (16,24), interpolation = cv2.INTER_NEAREST)

    # add img together x3 for total transform
    img = np.concatenate((img, img, img), axis=1)

    # flatten array
    img = img.flatten()

    # stringify for server
    count = 0
    transformSend = ""
    for ele in img:
        count += 1
        transformSend+=(" "+str(ele))

    # if you want to look at the numbers :)
    print (count)

    # uncomment this to send to server
    channel.push("input",{"body": transformSend})


def links():
    print ('links')
    transform('./links.jpg')

def headings():
    print ('headings')
    transform('./headings.jpg')

def images():
    print ('images')
    transform('./images.jpg')


def listen(function, keyword):
    print ('listening for' + keyword)
    recognition(function, keyword, True)

if __name__ == "__main__":
    t1 = Thread(target = listen(links,'links'))
    t1.start()
    t2 = Thread(target = listen(headings,'headings'))
    t2.start()
    t3 = Thread(target = listen(images, 'images'))
    t3.start()
