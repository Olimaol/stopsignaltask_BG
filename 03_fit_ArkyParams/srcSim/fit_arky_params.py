import numpy as np
import pylab as plt
import csv


def read_arky_data():
    file_rows = {}

    inj_current_Arky_pA = np.nan * np.ones(18)
    rate_Arky_Hz = np.nan * np.ones(18)

    #with open('data/Abdi_etal_2015_Fig7Fdata_ArkyProtoPosNeg.csv', 'rb') as csvfile:    # Python2
    with open('data/Abdi_etal_2015_Fig7Fdata_ArkyProtoPosNeg.csv', newline='') as csvfile:    # Python3
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            i_line = int(reader.line_num)
            file_rows[str(i_line)] = row
            line_data = file_rows[str(i_line)]
            if line_data[0] != 'Arky_data' and line_data[0] != 'X' and line_data[0] != '':
                inj_current_Arky_pA[i_line] = float(line_data[0])
                rate_Arky_Hz[i_line] = float(line_data[1])

    n_rows = len(file_rows) - 2 # First two rows are comments, not data


    sortind = np.argsort(inj_current_Arky_pA)
    inj_current_Arky_pA = np.round(inj_current_Arky_pA[sortind])
    rate_Arky_Hz = rate_Arky_Hz[sortind]

    #print("inj_current_Arky_pA = ", inj_current_Arky_pA)
    #print("rate_Arky_Hz = ", rate_Arky_Hz)

    return inj_current_Arky_pA, rate_Arky_Hz


def read_arky_data_Bogacz2016():
    # GPe Proto min. value of membrane potential durin autonomous firing (Fig. 3B): -66mV
    # GPe Arky min. value of membrane potential durin autonomous firing (Fig. 3B): -71mV    

    file_rows = {}

    inj_current_Arky_pA = np.nan * np.ones(32)
    rate_Arky_Hz = np.nan * np.ones(32)
    rate_Proto_Hz = np.nan * np.ones(32)    
    sem_Arky_Hz = np.nan * np.ones(32)    
    sem_Proto_Hz = np.nan * np.ones(32)    

    #with open('../experimental/Bogacz_etal_2016_S1data.csv', 'rb') as csvfile:    # Python2
    with open('../experimental/Bogacz_etal_2016_S1data.csv', newline='') as csvfile:    # Python3
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            i_line = int(reader.line_num)
            file_rows[str(i_line)] = row
            line_data = file_rows[str(i_line)]
            #if line_data[0] != 'Injected current (pA)':
            if i_line >= 3:
                inj_current_Arky_pA[i_line-3] = float(line_data[0].replace(',','.'))
                rate_Proto_Hz[i_line-3] = float(line_data[1].replace(',','.'))                                
                sem_Proto_Hz[i_line-3] = float(line_data[3].replace(',','.'))                                
                rate_Arky_Hz[i_line-3] = float(line_data[4].replace(',','.'))
                sem_Arky_Hz[i_line-3] = float(line_data[6].replace(',','.'))                

    i_inj150pA = np.nonzero(inj_current_Arky_pA == 150)[0][0]
    i_inj250pA = np.nonzero(inj_current_Arky_pA == 250)[0][0]    
    #print('i_inj150pA, i_inj250pA = ', i_inj150pA, i_inj250pA)

    n_rows = len(file_rows) - 2 # First two rows are comments, not data

    #print("inj_current_Arky_pA = ", inj_current_Arky_pA)
    #print("rate_Arky_Hz = ", rate_Arky_Hz)
    #print("rate_Proto_Hz = ", rate_Proto_Hz)    

    #return inj_current_Arky_pA, rate_Arky_Hz, sem_Arky_Hz, rate_Proto_Hz, sem_Proto_Hz
    return inj_current_Arky_pA[0:i_inj250pA], rate_Arky_Hz[0:i_inj250pA], sem_Arky_Hz[0:i_inj250pA], rate_Proto_Hz[0:i_inj250pA], sem_Proto_Hz[0:i_inj250pA]    

if __name__ == '__main__':
    plt.ion()        
    inj_current_Arky_pA, rate_Arky_Hz = read_arky_data()
    plt.plot(inj_current_Arky_pA, rate_Arky_Hz, '.-', label='Arky (Abdi et al.)')    
    inj_current_Arky_pA, rate_Arky_Hz, sem_Arky_Hz, rate_Proto_Hz, sem_Proto_Hz = read_arky_data_Bogacz2016()    
    #plt.plot(inj_current_Arky_pA, rate_Arky_Hz, '.-', label='Arky (Abdi et al.)')
    plt.errorbar(inj_current_Arky_pA, rate_Arky_Hz, yerr= sem_Arky_Hz, label='Arky (Bogacz et al.)')        
    #plt.plot(inj_current_Arky_pA, rate_Proto_Hz, '.-', label='Proto (Abdi et al.)')    
    plt.errorbar(inj_current_Arky_pA, rate_Proto_Hz, yerr= sem_Proto_Hz, label='Proto (Bogacz et al.)')            
    plt.xlabel('Injected current [nA]')
    plt.ylabel('Firing rate [spk/sec]')
    plt.legend()
    plt.savefig('plots/read_Prot_Arky_data_Bogacz2016.png')
    plt.ioff()
    plt.show()
