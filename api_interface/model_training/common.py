import os

def get_tmp_dir():
    now_dir = os.getcwd()
    tmp = os.path.join(now_dir, "TEMP")
    os.makedirs(tmp, exist_ok=True)
    return tmp