import cv2, time
import numpy as np
from occamy import Socket
import threading
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
    img = cv2.resize(img, (48,24), interpolation = cv2.INTER_NEAREST)

    # flatten array
    img = img.flatten()

    # stringify for server
    transformSend = ""
    for ele in img:
        count += 1
        transformSend+=(" "+str(ele))

    # if you want to look at the numbers :)
    print (transformSend)

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
    t1 = threading.Thread(target = listen, args=(links,'links'))
    t2 = threading.Thread(target = listen, args=(headings,'headings'))
    t3 = threading.Thread(target = listen, args=(images, 'images'))
    t1.start()
    time.sleep(1)
    t2.start()
    time.sleep(1)
    t3.start()
