import cv2
import sys
import numpy as np
import datetime
import os
import lmdb
import struct

lmdb_path = "../images"
lmdb_path2 = "../images2"
#writer = LmdbWriter(os.path.join(lmdb_path,lmdb_name), map_size=int(6e10))
env = lmdb.open(lmdb_path, max_dbs=8, map_size=int(1e12), readonly=True, lock = False)
env2 = lmdb.open(lmdb_path2, max_dbs=8, map_size=int(5e10), writemap=True)
env2 = env
env.close()
env2.close()
