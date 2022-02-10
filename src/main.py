import argparse
import pandas as pd
from utils import *
from configuration import config_algo_ds
from prophet import Proph
from arimax import Arimax
from sarimax import Sarimax
from ast import literal_eval as make_tuple


# INPUTS
my_config = config_algo_ds()
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str,
                    help="Enter the model you want to use", required=True)
parser.add_argument("--order", type=str,
                    help="Enter the order you want to fit on the ARIMAX model p,d,q , example : 1,1,1", required=False, default="1, 1, 1")
parser.add_argument("--sorder", type=str, help="Enter the seasoal order",
                    required=False, default="1, 1, 1, 4")
parser.add_argument(
    "--k", type=str, help="The number of points we make the predictions on", required=False, default="1000")

PATH = my_config['PATH']['PATH']
PATH_INOUT = my_config['PATH']['DATA_PATH_FULL']
PATH_OUT = my_config['PATH']['DATA_PATH_OUT']
PATH_IN = my_config['PATH']['DATA_PATH_IN']
TEST_SIZE = float(my_config["PARAMETERS"]["TEST_SIZE"])
NUM_WORKER = int(my_config['PARAMETERS']['NUM_WORKER'])


if __name__ == '__main__':

    # Read instructions from argparse
    args = parser.parse_args()
    model_type = args.model
    arima_order = make_tuple(args.order)
    seasonal_order = make_tuple(args.sorder)
    k = int(args.k)

    # Print
    print("-------------")
    print("You will train a {} on cpu".format(model_type))
    print("-------------")

    # Print
    print("-------------")
    print(f"You will read {PATH} {PATH_INOUT}.")
    print("-------------")

    df = pd.read_csv(PATH + PATH_INOUT)

    # 'RES11_BASE','PRO1_BASE', 'RES2_HC', 'RES2_HP', 'PRO2_HC', 'PRO2_HP']
    targets = ['RES1_BASE']

    if model_type == "Prophet":
        for target in targets:
            print("-------------")
            print(f"A prophet model is running on {target}.")
            print("-------------")

            Proph(target, df)

            print("-------------")
            print(f"Prophet model is done with {target}.")
            print("-------------")

    elif model_type == "ARIMAX":
        for target in targets:
            print("-------------")
            print(f"An ARIMAX model is running on {target}.")
            print("-------------")

            Arimax(target, df, arima_order, k)

            print("-------------")
            print(f"Arimax model is done with {target}.")
            print("-------------")
    elif model_type == "SARIMAX":
        for target in targets:
            print("-------------")
            print(f"A SARIMAX model is running on {target}.")
            print("-------------")

            Sarimax(target, df, arima_order, seasonal_order, k)

            print("-------------")
            print(f"Sarimax model is done with {target}.")
            print("-------------")

    else:
        print("Nothing to do... Please enter the followong command right after main.py: --model=THE-MODEL-YOU-WANT-TO-USE")
     # Print
    print("-------------")
    print(f"Check you mail. Result will be sended automatically.")
    print("-------------")
