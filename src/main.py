from matplotlib import markers
from tqdm import tqdm

import os
import argparse
import pandas as pd
from utils import *
from configuration import config_algo_ds
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mailsender import send_csv_by_mail


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
    print(f"You will read {PATH} {PATH_INOUT}.")
    print("-------------")

    df = pd.read_csv(PATH+PATH_INOUT)

    print("-------------")
    print(f"Preprocessing is on going...")
    print("-------------")

    X,y = preprocessing_tuned(df)


    print("-------------")
    print(f"Preprocessing is finished, start training...")
    print("-------------")
    X_train, X_test, y_train, y_test  = train_test_split( X, y, test_size=TEST_SIZE,shuffle=False)

    print(y_test.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(X_train.shape)

    targets = ['RES1_BASE', 'RES11_BASE','PRO1_BASE', 'RES2_HC', 'RES2_HP', 'PRO2_HC', 'PRO2_HP']
    
    if model_type == "ARIMAX" : 
        for target in targets :
            best_cfg,best_score=arimax_grid_search(y_train[target],y_test[target],np.array([3,4,5]),np.arange(5),np.array([3,4,5]),X_train,X_test)
            error, predictions=evaluate_arimax_model(y_train[target],y_test[target], best_cfg, X_train, X_test)
            plt.figure(figsize=(12,8))
            plt.plot(predictions,c="blue",label="Predicted",markers="+")
            plt.plot(y_test[target],c="green",label="True",markers="x")
            plt.grid()
            plt.legend()
            plt.xlabel("Time")
            plt.ylabel(f"{target}")
            plt.title(f"Predicted vs True value for {target} with {error} error")
            plt.savefig(f"../outputs/{target}.png")
        send_csv_by_mail()
  



   


