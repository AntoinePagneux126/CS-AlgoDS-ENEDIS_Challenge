import argparse
import pandas as pd
from utils import *
from configuration import config_algo_ds
from prophet import Proph

# exemple to run:   python3 main.py --mode=train --model=cnn
#                   python3 main.py --mode=train --model=resnet50
#                   python3 main.py --mode=test --model=cnn --number=1

## INPUTS
my_config = config_algo_ds()
parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, help="Enter the mode you want to use (train/test)", required=False, default="train")
parser.add_argument("--model", type=str, help="Enter the model you want to use", required=True)



PATH            = my_config['PATH']['PATH']
PATH_INOUT      = my_config['PATH']['DATA_PATH_FULL']
PATH_OUT        = my_config['PATH']['DATA_PATH_OUT']
PATH_IN         = my_config['PATH']['DATA_PATH_IN']
TEST_SIZE       = float(my_config["PARAMETERS"]["TEST_SIZE"])
NUM_WORKER      = int(my_config['PARAMETERS']['NUM_WORKER'])


if __name__ == '__main__':
    
    # Read instructions from argparse
    args        = parser.parse_args()
    mode        = args.mode
    model_type  = args.model


    # Print 
    print("-------------")
    print("You will {} a {} on cpu".format(mode, model_type))
    print("-------------")



    # Print 
    print("-------------")
    print(f"You will read {PATH} {PATH_INOUT}.")
    print("-------------")

    df = pd.read_csv(PATH+PATH_INOUT)

    targets = ['RES1_BASE']# 'RES11_BASE','PRO1_BASE', 'RES2_HC', 'RES2_HP', 'PRO2_HC', 'PRO2_HP']

    if model_type == "Prophet" :
        for target in targets :
            print("-------------")
            print(f"A prophet model is running on {target}.")
            print("-------------")

            Proph(target,df)

            print("-------------")
            print(f"Prophet model is done with {target}.")
            print("-------------")

    elif model_type == "ARIMAX" : 
        pass
    else : 
        pass
     # Print 
    print("-------------")
    print(f"Check you mail. Result will be sended automatically.")
    print("-------------")