import sys
import time
import cofilter
import datasetInput
from original.main import evaluate
import ConfigParser
from numpy import *

def getConfig(ConfigPath):
    config = ConfigParser.ConfigParser()
    config.read(ConfigPath)
    return


def getDictConf(ExpConfig, section):
    conf_dict = dict()
    try:
        para = ExpConfig.items(section)
    except Exception as e:
        print('Error conf item: ' + section)
        return conf_dict

    for item in para:
        conf_dict[item[0]] = item[1]

    return conf_dict

def main():
    ExpConfig = getConfig('../recommand.conf')
    user_num = int(ExpConfig.get('Dataset', 'user_num'))
    item_num = int(ExpConfig.get('Dataset', 'item_num'))
    para = getDictConf(ExpConfig, 'Parameters')
    res_file = ExpConfig.get('Files', 'output')
    data_file = ExpConfig.get('Files', 'train')
    test_file = ExpConfig.get('Files', 'test')
    # pic_file = ExpConfig.get('Files', 'pic')

    if 'clus_k' not in para:
        print('Error Parameters Conf: lost cluster k')
        return -1
    clus_k = int(para['clus_k'])

    if 'top_k' not in para:
        print('Error Parameters Conf: lost top k')
        return -1
    top_k = int(para['top_k'])

    if 'rec_l' not in para:
        print('Error Parameters Conf: lost rec l')
        return -1

    rec_l = int(para['rec_l'])
    data_arr = zeros((user_num, item_num))
    res_arr = zeros((user_num, item_num))
    test_arr = zeros((user_num, item_num))

    datasetInput.load(data_file, data_arr)
    datasetInput.load(test_file, test_arr)

    pic_axis = list()
    rmse_list = list()
    mae_list = list()

    outfile_name = res_file + time.strftime(".%Y%m%d-%H%I%S",
                                            time.localtime(time.time()))
    outfile = open(outfile_name, 'w')

    for ti in range(top_k, 25, 10):
        outfile.write("exp condition is : \n")
        para['top_k'] = ti

        for item in para:
            outfile.write(item + " : " + str(para[item]) + "\n")
        res_arr, simi_arr = cofilter.run(para, data_arr, res_arr)
        savetxt(outfile_name + '.res_arr_' + str(ti), res_arr)
        savetxt(outfile_name + '.simi_arr_' + str(ti), simi_arr)
        res_rmse, res_mae = evaluate.cal(res_arr, test_arr, outfile)
        pic_axis.append(ti)
        mae_list.append(res_mae)
        rmse_list.append(res_rmse)

    # res_pic.draw(pic_file + '.mae.png', pic_axis, mae_list, 'clus_k', 'mae')
    # res_pic.draw(pic_file + '.rmse.png', pic_axis, rmse_list, 'clus_k', 'rmse')

    outfile.close()

if __name__ == '__main__':
    main()
