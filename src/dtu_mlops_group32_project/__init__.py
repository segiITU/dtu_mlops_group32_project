import os

_FILE_ROOT = os.path.dirname(__file__)  # root of folder
_SRC_ROOT = os.path.dirname(_FILE_ROOT)  # root of src
_PROJECT_ROOT = os.path.dirname(_SRC_ROOT)  # root of project
_PATH_DATA = os.path.join(_PROJECT_ROOT, "data")  # root of data