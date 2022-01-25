from tqdm import tqdm

import os
import argparse
import pandas as pd
from utils import *
from configuration import config_algo_ds


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
NUM_WORKER      = int(my_config['PARAMETERS']['NUM_WORKER'])


if __name__ == '__main__':
    
    # Read instructions from argparse
    args        = parser.parse_args()
    mode        = args.mode
    model_type  = args.model


    # Print 
    print("-------------")
    print(f"You will read {PATH} {PATH_INOUT}.")
    print("-------------")

    df = pd.read_csv(PATH+PATH_INOUT)

    print("-------------")
    print(f"Preprocessing is on going...")
    print("-------------")

    X,y = preprocessing(df)

    print("-------------")
    print(f"Preprocessing is finished, start training...")
    print("-------------")


   


