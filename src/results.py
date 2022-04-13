# This file was used to create boxplots for thesis report


import matplotlib.pyplot as plt
import pandas as pd


def load_dataframe(filename="data/F1_scores_29_6_2021.csv"):
    df = pd.read_csv(filename, index_col=0)
    return df


def store_dataframe(name_list, data_lists, filename="data/F1_scores_29_6_2021.csv", ):
    # dict = {"name": name_list, "score": score_list, "upper": upper_bound_score_list, "analyst": analyst_score_list}
    # df = pd.dataframe(dict)
    df = pd.DataFrame(data=data_lists, index=name_list, columns=["set", "mix", "donors", "score", "upper", "analyst"])
    df.to_csv(filename, index=False)
    return df


def make_boxplot(dataframe, columns, group_by):
    boxplot = dataframe.boxplot(column=columns, by=group_by)
    # plt.ylim([0,1])
    plt.show()
