paramsA = {}

paramsA['loadFolder'] = 'longertestSim'

### SPECIFIC PARAMETERS FOR 'STR'
paramsA['STR_fsize'] = 15
paramsA['STR_figC_tmax'] = 200
paramsA['STR_figC_tmin'] = -50
paramsA['STR_figC_xticks'] = [0,0.1,0.2]
paramsA['STR_figAB_tmax'] = 350
paramsA['STR_figAB_tmin'] = -200
paramsA['STR_figAB_xticks'] = [-0.2, 0, 0.3]# in seconds
paramsA['STR_figABC_ymin'] = -2.2
paramsA['STR_figABC_ymax'] = 3.6
paramsA['STR_figABC_yticks'] = range(-2,4)

### SPECIFIC PARAMETERS FOR 'extra'
paramsA['Fig7_maxRates'] = {'GPeArky':60, 'StrD1':105, 'StrD2':70, 'STN':80, 'cortexGo':320, 'GPeCp':75, 'GPeProto':50, 'SNr':130, 'Thal':45, 'cortexStop':430, 'StrFSI':50, 'IntegratorGo':0.25}
