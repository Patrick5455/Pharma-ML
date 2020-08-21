import os


class Config(object):
    # add configuration variables
    SECRET_KEY = os.environ.get("SECRET_KEY") or "you_will_never_guess"
