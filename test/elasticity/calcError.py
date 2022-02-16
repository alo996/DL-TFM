import numpy as np
import pandas as pd
import os
from scipy.io import loadmat
from add_noise import addNoise
from errorTrac import errorTrac


def calcError(path):
    errors = []
    S = 160
    y_moduli = [2500, 5000, 10000, 20000, 400000]
    noise = 0.00765
    for nr in range(18, 56):
        name = f'MLData00{nr}.mat'.format
        path = os.path.join(path, name)
        if os.path.isfile(path):
            cell = [nr]
            trac_file = loadmat(path)
            brdx = trac_file['brdx']
            brdy = trac_file['brdy']
            tracGT = trac_file['tracGT']
            for i in y_moduli:
                name = f'MLData00{nr}-{i}.mat'.format(nr=nr, i=i)
                path = os.path.join(path, name)
                if os.path.isfile(path):
                    dspl_file = loadmat(path)
                    dspl = dspl_file['dspl']
                    trac = predictTrac(dspl, i)
                    err = errorTrac(trac, tracGT, brdx, brdy)
                    cell.append(err)

                    dspl = addNoise(dspl, noise)
                    trac = predictTrac(dspl, i)
                    err = errorTrac(trac, tracGT, brdx, brdy)
                    cell.append(err)
                else:
                    continue
            errors.append(cell)
        else:
            continue
    df = pd.DataFrame(errors, index=['first', 'second'],
                      columns=['File ID', '2,500Pa', '2,500Pa N', '5,000Pa', '5,000Pa N', '10,000Pa', '10,000Pa N',
                               '20,000Pa', '20,000Pa N', '40,000Pa', '40,000Pa N'])
    return df
