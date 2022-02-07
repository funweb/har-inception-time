import yaml
import os
import time
import shutil
import emoji  # https://carpedm20.github.io/emoji/all.html


def clock(func):
    """
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    interval_time = (
            datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S.%f") -
            datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S.%f")).total_seconds()

    """

    def clocked(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print("{}: {}".format(func.__name__, end - start))
        return result

    return clocked


@clock
def load_config(config_path="tools/configure/config.yaml"):
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as fr:
            return yaml.load(fr, Loader=yaml.FullLoader)
    else:
        assert False, "config file does not exits: {}".format(os.path.join(os.getcwd(), config_path))


def create_folder(path='./new', remake=False):
    # Create folder
    if not os.path.exists(path):
        print('Create subdir directory: %s...' % (path))
        time.sleep(3)
        os.makedirs(path)
    elif remake:
        shutil.rmtree(path)  # delete output folder
        os.makedirs(path)


# Merge the two dictionaries, that is, the parameters required by the algorithm
def Merge(dict_config, dict_config_cus):
    dict_config.update(dict_config_cus)
    for k in dict_config:  # 保证值不为空, 也就是保证参数的有效性
        assert dict_config[k] != "", "Please set value for: {}".format(k)
    return dict_config


def reTrain(p):
    if os.path.exists(p):
        shutil.rmtree(p)


import urllib, json, os, ipykernel, ntpath
from notebook import notebookapp as app


def lab_or_notebook():
    length = len(list(app.list_running_servers()))
    if length:
        return "notebook"
    else:
        return "lab"


def ipy_nb_name(token_lists):
    """ Returns the short name of the notebook w/o .ipynb
        or get a FileNotFoundError exception if it cannot be determined
        NOTE: works only when the security is token-based or there is also no password
    """

    if lab_or_notebook() == "lab":
        from jupyter_server import serverapp as app
    else:
        from notebook import notebookapp as app
    #         from jupyter_server import serverapp as app

    connection_file = os.path.basename(ipykernel.get_connection_file())
    kernel_id = connection_file.split('-', 1)[1].split('.')[0]

    #     from notebook import notebookapp as app
    for srv in app.list_running_servers():
        for token in token_lists:
            srv['token'] = token

            try:
                # print(token)
                if srv['token'] == '' and not srv['password']:  # No token and no password, ahem...
                    req = urllib.request.urlopen(srv['url'] + 'api/sessions')
                    print('no token or password')
                else:
                    req = urllib.request.urlopen(srv['url'] + 'api/sessions?token=' + srv['token'])
            except:
                pass
                # print("Token is error")

        sessions = json.load(req)

        for sess in sessions:
            if sess['kernel']['id'] == kernel_id:
                nb_path = sess['notebook']['path']
                return ntpath.basename(nb_path).replace('.ipynb', '')  # handles any OS

    raise FileNotFoundError("Can't identify the notebook name, Please check [token]")


def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


if __name__ == '__main__':
    # print(load_config("configure/config.yaml"))
    print(colorstr("bold", "bright_red", 222))
    print(colorstr("bright_red", 222))
