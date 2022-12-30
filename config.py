from os import environ as env
import multiprocessing

# Gunicorn config
bind = ":" + str(8000)
workers = multiprocessing.cpu_count() * 2 + 1
threads = 2 * multiprocessing.cpu_count()