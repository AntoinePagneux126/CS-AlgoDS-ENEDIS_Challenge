import pandas as pd 


def merge(file_in="../dataset/inputs.csv",file_out="../dataset/outputs.csv"): 
    """[merge inputs and outputs datasets]

    Args:
        file_in (str, optional): [path to inputs csv file]. Defaults to "../dataset/inputs.csv".
        file_out (str, optional): [path to outputs csv file]. Defaults to "../dataset/outputs.csv".
    """
    if not isinstance(file_in,str) or not isinstance(file_out,str):
        raise Exception("file must be string which represent path to csv files")

    df_in  = pd.read_csv(file_in)
    df_out = pd.read_csv(file_out)
    df = pd.merge(df_in,df_out,how="inner",
                    on = ['IDS', 'Horodate_UTC', 'Horodate', 'Mois'],
                    sort=True,
                    copy=True,
                    indicator=False,
                    validate=None,
                )
    df.to_csv("../dataset/inout.csv",index=False)


if __name__ == "__main__" :
    merge()