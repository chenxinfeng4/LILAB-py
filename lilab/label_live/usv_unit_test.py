import picklerpc
from lilab.label_live.usv1_realtime_yolo import main
import threading
import time
from lilab.label_live.socketServer import start_socketserver_background

# process = threading.Thread(target=main)
# process.start()

# rpcclient = picklerpc.PickleRPCClient(('localhost', 8092))

# while True:
#     time.sleep(0.5)
#     dt = rpcclient.usv_queue_get()
#     print(dt)

start_socketserver_background()

main()
