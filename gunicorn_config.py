# Gunicorn configuration for Render free tier
# Optimized for low memory usage

import multiprocessing

# Workers (use 1 for free tier to save memory)
workers = 1

# Worker class
worker_class = "sync"

# Timeout (increase for model loading)
timeout = 120  # 2 minutes (default is 30s)

# Memory optimization
max_requests = 1000
max_requests_jitter = 50

# Preload app (loads model once, shared by workers)
preload_app = True

# Keep alive
keepalive = 5

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

