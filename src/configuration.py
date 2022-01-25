import configparser

CONFIG_PATH = "../config/"
FILE_NAME = "config.ini"


def config_algo_ds():
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH + FILE_NAME)
    return config


if __name__ == "__main__":
    config = config_algo_ds()
    print("Path: ", config['PATH']['DATA_PATH'])
    print("Learing Rate: ", config['PARAMETERS']['NUM_WORKER'])
