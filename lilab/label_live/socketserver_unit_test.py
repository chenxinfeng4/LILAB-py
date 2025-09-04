import picklerpc

rpcclient = picklerpc.PickleRPCClient(("localhost", 8092))

rpcclient.usv_queue_put(4)
rpcclient.usv_queue_put(4)
rpcclient.usv_queue_get()

rpcclient.bhv_queue_put(3)
rpcclient.bhv_queue_get()
