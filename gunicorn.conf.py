import os
os.makedirs('logs', exist_ok=True)

open('logs/error.log', 'a').close()
open('logs/access.log', 'a').close()

bind = '0.0.0.0:5000'
workers = 1
threads = 4
timeout = 600
errorlog = 'logs/error.log'
accesslog = 'logs/access.log'
loglevel = 'info'

# Note: We cannot use multiple workers together with preload_app in gunicorn to
# boost memory and compute efficiency, for bad compatibility with CUDA.
# set workers -> multi-process (not recommended, unless you have enough computing resources)
# set threads -> multi-thread
