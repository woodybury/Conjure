from occamy import Socket

def join():
    socket = Socket("ws://dlevs.me:4000/socket")
    socket.connect()

    channel = socket.channel("room:lobby", {})
    channel.on("connect", print ('Im in'))
    #channel.on("touch", print("no("))
    #channel.on("touch", lambda msg, x: print("> {}".format(msg["body"])))

    # channel.join()

    return channel
