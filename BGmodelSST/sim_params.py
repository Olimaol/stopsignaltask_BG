import numpy as np
import csv
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))

curr_ID = 8007

def readParams(csvPath,param_ID,integerParams):
    """
    read all parameters from specified csv file

    csvPath : str
        path to the .csv file 
    param_ID : int
        specifies which column in the csv file is used
    integerParams : list of str
        list of parameter names whose values should be integer instead of float
    """

    if sys.version_info[0]<3:
        csvfile = open(csvPath, 'rb')# Python 2
    else:
        csvfile = open(csvPath, newline='')# Python 3

    params = {}
    reader = csv.reader(csvfile, delimiter=',')
    fileRows = []
    idx = -1
    ### check if param_ID is in the .csv file
    for row in reader:
        fileRows.append(row)
        if 'general_id'==row[0] and True in [param_ID == int(float(row[i])) for i in range(1,len(row))]:
            idx = [param_ID == int(float(row[i])) for i in range(1,len(row))].index(True)+1
        elif 'general_id'==row[0]:
            print('No Parameters available for given parameter ID! (file '+__file__+')')
            quit()
    if idx==-1:
        print('No general_id in parameter file!')
        quit()
    ### read the column corresponding to param_ID
    for row in fileRows:
        if '###' in row[0]: continue
        if row[0] in integerParams:
            params[row[0]] = int(float(row[idx]))
        else:
            params[row[0]] = float(row[idx])

    csvfile.close()
    return params


### define parameters with other datatype as float
integerParams = ['general_populationSize', 'GPeArkyCopy_On', 'threads', 'general_id']

### get parameters from csv file
csvfilePath = dir_path+'/new_params.csv'
params = readParams(csvfilePath,curr_ID,integerParams)

#params['tau_syn_factor'] = 1.0
#params['TrialType'] = 2.0
#params['ratesProto1vs2'] = 0.7
#params['t_StopDuration'] = 500.0
params['Stop_Proto'] = 0.0
params['Arky_CortexGo'] = 0.0
params['Arky_Int'] = -10000.0
params['Gpe_Proto2_Int'] = -10000.0
params['Int_init'] = -10000.0
