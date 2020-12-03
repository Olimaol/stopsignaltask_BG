import numpy as np
import pylab as plt
import glob
import matplotlib as mtl
import os

### functions
def sizedArrow(x0, y0, x1, y1, width, color, style):
    sizedArrowStyle = mtl.patches.ArrowStyle.Curve()
    arrow = mtl.patches.FancyArrowPatch((x0,y0), (x1,y1), connectionstyle='arc3,rad=0', shrinkA = 0, shrinkB = 0,clip_on=False, **dict(arrowstyle=sizedArrowStyle, linewidth=width, color=color, linestyle=style, transform=ax.transAxes))
    ax.add_patch(arrow)

### Parameters
p = {}
p['loadFolder'] = 'testSim'
p['plotFolder'] = '../results/'+p['loadFolder']
p['networkIDs'] = [1,2,3,4,5]
p['paramID']    = 8007
p['oneFile']    = False
fig = {}
fig['x']                   = 85
fig['y']                   = 85
fig['dpi']                 = 300
fig['linewidth']           = 1.5
fig['legend_names']        = {'Arky' : 'GPe-Arky', 'STN' : 'STN', 'Proto2' : 'GPe-Cp', 'None' : 'None'}
fig['colors']              = {'Arky' : np.array([0,255,255])/255., 'STN' : np.array([219,180,12])/255., 'Proto2' : np.array([173,129,80])/255., 'None' : 'black'}
fig['linestyles']          = {'Arky' : 'dashed', 'STN' : 'dashdot', 'Proto2' : 'dotted', 'None' : 'solid'}
fig['xtickVals']           = [0,100,200,300,400,500]
fig['xtickNames']          = [0,100,200,300,400,500]
fig['ytickVals']           = [0,0.2,0.4,0.6,0.8,1.0]
fig['ytickNames']          = [0,20,40,60,80,100]
fig['xlim']                = [0,500]
fig['xlabel']              = 'SSD [ms]'
fig['ylabel']              = '% correct Stop trials'
fig['bottom']              = 0.124
fig['top']                 = 0.99
fig['right']               = 0.965
fig['left']                = 0.163
fig['legend_left']         = 0.62
fig['legend_bottom']       = 0.63
fig['legend_width']        = 0.3
fig['legend_height']       = 0.3
fig['legend_pad']          = 0.015
fig['legend_order']        = ['None', 'STN', 'Arky', 'Proto2']
fig['legend_freespace']    = 0.02
fig['legend_betweenspace'] = fig['legend_height']/5.*1.2
fig['legend_linelength']   = 0.1
fig['legend_title']        = 'lesion:'
font = {}
font["axLabel"] = {'fontsize': 11, 'fontweight' : 'normal'}
font["axTicks"] = {'labelsize': 9}
font["legend1"]  = {'fontsize': 9, 'fontweight' : 'normal'}
font["legend2"]  = {'size': 9, 'weight' : 'normal'}

### FOLDER CREATION
dataDir=p['plotFolder']
try:
    os.makedirs(dataDir)
except:
    if os.path.isdir(dataDir):
        print(dataDir+' already exists')
    else:
        print('could not create '+dataDir+' folder')
        quit()

### Results
stoppingPerformance = []
lessionList = []
t_SSD_List = []

### Load
if p['oneFile']:
    for i_netw_rep in p['networkIDs']:
        container=np.load('../data/'+p['loadFolder']+'/SSD_variation_stoppingPerformance_paramId'+str(p['paramID'])+'_netID'+str(i_netw_rep)+'.npz')
        stoppingPerformance.append(container['stoppingPerformance'])
        lessionList=container['lessionList']
        t_SSD_List=container['t_SSD_List']
    stoppingPerformance = np.array(stoppingPerformance)
else:
### single files
    files=glob.glob('../data/'+p['loadFolder']+'/SSD_variation_stoppingPerformance*_paramId'+str(p['paramID'])+'*')
    for f in files:
        if 'Arky' in f or 'STN' in f or 'Proto2' in f or 'None' in f:
            stoppingPerformance.append(np.load(f))
            lessionList.append(f.split('_')[-4][19:])
            t_SSD_List.append(f.split('_')[-3])
    data=np.array([stoppingPerformance,lessionList,t_SSD_List])

### Analysis
if p['oneFile']:
    stoppingPerformance_mean = np.mean(stoppingPerformance,0)
    stoppingPerformance_sd = np.std(stoppingPerformance,0)
else:
    plotData = {}
    for lessionIdx, lession in enumerate(np.unique(data[1])):
        plotData[lession+'_mean'] = []
        plotData[lession+'_sd']   = []
        for timeIdx, time in enumerate(np.sort(np.unique(data[2].astype(int)))):
            mask1 = data[1]==lession
            mask2 = data[2].astype(int)==time
            mask = mask1*mask2
            if True in mask:
                plotData[lession+'_mean'].append(np.mean(data[0,mask].astype(float)))
                plotData[lession+'_sd'].append(np.std(data[0,mask].astype(float)))



### Plots  
plt.figure(figsize=(fig['x']/25.4, fig['y']/25.4), dpi=fig['dpi'])
ax = plt.subplot(111)
if p['oneFile']:
    for lessionIdx, lession in enumerate(lessionList):
        ax.errorbar(t_SSD_List, stoppingPerformance_mean[lessionIdx], yerr=stoppingPerformance_sd[lessionIdx], label=str(lession), alpha=0.7)
else:
    line = {}
    for lessionIdx, lession in enumerate(np.unique(data[1])):
        mean = plotData[lession+'_mean']
        sd = plotData[lession+'_sd']
        time = np.sort(np.unique(data[2].astype(int)))[:len(mean)]
        #ax.errorbar(time, mean, yerr=sd, label=str(lession), alpha=0.7, linewidth=fig['linewidth'], color=fig['colors'][lession], linestyle=fig['linestyles'][lession])
        ax.fill_between(time, np.array(mean)-np.array(sd), np.array(mean)+np.array(sd), color=fig['colors'][lession], alpha=0.4, linewidth=0)
        line[lession], = ax.plot(time, mean, label=fig['legend_names'][lession], linewidth=fig['linewidth'], color=fig['colors'][lession], linestyle=fig['linestyles'][lession])
            
        #ax.scatter(data[2,data[1]==lession].astype(int), data[0,data[1]==lession].astype(float), label=str(lession), linewidths=0, s=1)
plt.xlim(fig['xlim'][0],fig['xlim'][1])
plt.xticks(fig['xtickVals'],fig['xtickNames'])
plt.yticks(fig['ytickVals'],fig['ytickNames'])
ax.tick_params(axis='both', which='both', **font["axTicks"])
ax.set_xlabel(fig['xlabel'], **font["axLabel"])
ax.set_ylabel(fig['ylabel'], **font["axLabel"])
plt.subplots_adjust(bottom = fig['bottom'], top = fig['top'], right = fig['right'], left = fig['left'])
### legend
"""legendField=mtl.patches.FancyBboxPatch(xy=(fig['legend_left'],fig['legend_bottom']),width=fig['legend_width'],height=fig['legend_height'],boxstyle=mtl.patches.BoxStyle.Round(pad=fig['legend_pad']),bbox_transmuter=None,mutation_scale=1,mutation_aspect=None,transform=ax.transAxes,**dict(linewidth=2, fc='w',ec='k',clip_on=False))
ax.add_patch(legendField)
for Idx in range(len(fig['legend_order'])):
    lession = fig['legend_order'][Idx]
    x0 ,y0 = fig['legend_left']+fig['legend_freespace'], fig['legend_bottom']+fig['legend_height']/2.+(fig['legend_betweenspace']/2.)*(3-2*Idx)
    x1 ,y1 = x0+fig['legend_linelength'], y0
    sizedArrow(x0, y0, x1, y1, fig['linewidth'], fig['colors'][lession], fig['linestyles'][lession])
    plt.text(x1+fig['legend_freespace'],y1,fig['legend_names'][lession],ha='left',va='center',transform=ax.transAxes, **font["legend1"])"""
leg=ax.legend((line[lession] for lession in fig['legend_order']), (fig['legend_names'][lession] for lession in fig['legend_order']), prop=font["legend2"], title=fig['legend_title'])
leg._legend_box.align = "left"

### save
plt.savefig(p['plotFolder']+'/SSD_variation_paramID'+str(p['paramID'])+'.svg')
























