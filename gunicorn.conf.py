import os
os.makedirs('logs', exist_ok=True)

open('logs/error.log', 'a').close()
open('logs/access.log', 'a').close()

bind = '0.0.0.0:5000'
worker = 1
timeout = 600
errorlog = 'logs/error.log'
accesslog = 'logs/access.log'
loglevel = 'info'
