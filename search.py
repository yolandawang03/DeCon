#coding = utf-8

import time
import os


def ReadAllTriples(files):
    dict = {}

    for f in files:
        file = open(f, "r")
        for line in file:
            list = line.split(" ")

            if list[0] in dict.keys():
                if list[1] in dict.get(list[0]).keys():
                    dict.get(list[0]).get(list[1]).append(list[2].strip('\n'))
                else:
                    dict.get(list[0])[list[1]] = [list[2].strip('\n')]
            else:
                dict[list[0]] = {list[1]:[list[2].strip('\n')]}

        # for key in dict.keys():
        #     print(key+' : ',dict[k])
        file.close()

    return dict