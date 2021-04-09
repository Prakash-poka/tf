https://drive.google.com/file/d/1otz6JDvanbLxFCnPANrS_F3tDJncdJAA/view?usp=sharing
https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbmk5cnZYRFg5VXRiMG1NZTNmTHZubXVWd1BGZ3xBQ3Jtc0trYmtUV3BIeW5rakd1LU9rSzhiVjdaejJqSTI3c043ZnBJa1dpTWpTWEtSbVZOUnJhcndENzBMZm5XOV8tc2tOY0VITHRhTGQ4UEQ4T05ESXR1aGlMYzJFQUdQTTJVRWZtQmtjNHBNLU1DN1B3a1QySQ&q=https%3A%2F%2Fstorage.googleapis.com%2Fdownload.tensorflow.org%2Fmodels%2Ftflite%2Fcoco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
import pyaudio
import struct
import matplotlib.pyplot as plt
import numpy as np
mic = pyaudio.PyAudio()
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 20000
fin=[]
CHUNK = int(RATE/20)
stream = mic.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, output=True, frames_per_buffer=CHUNK)
'''fig, ax = plt.subplots(figsize=(14,6))
x = np.arange(0, 2 * CHUNK, 2)
ax.set_ylim(-200, 200)
ax.set_xlim(0, CHUNK) #make sure our x axis matched our chunk size
line, = ax.plot(x, np.random.rand(CHUNK))'''
while True:
    data = stream.read(CHUNK)
    data = np.frombuffer(data, np.int16)
    '''line.set_ydata(data)
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.01)'''
###############################################################
    mic_sens_dBV = -47.0 # mic sensitivity in dBV + any gain
    mic_sens_corr = np.power(10.0,mic_sens_dBV/20.0) # calculate mic sensitivity conversion factor
    
    # (USB=5V, so 15 bits are used (the 16th for negatives)) and the manufacturer microphone sensitivity corrections
    #data = ((data/np.power(2.0,15))*5.25)*(mic_sens_corr) 
    
    # compute FFT parameters
    f_vec = RATE*np.arange(CHUNK/2)/CHUNK # frequency vector based on window size and sample rate
    mic_low_freq = 100 # low frequency response of the mic (mine in this case is 100 Hz)
    low_freq_loc = np.argmin(np.abs(f_vec-mic_low_freq))
    fft_data = (np.abs(np.fft.fft(data))[0:int(np.floor(CHUNK/2))])/CHUNK
    fft_data[1:] = 2*fft_data[1:]
    
    fft_data=list(map(int,fft_data))
    fin.extend(fft_data)
    print(fin)

    # plot
    '''fig, ax = plt.subplots(figsize=(14,6))
    x = np.arange(0, 2 * CHUNK, 2)
    ax.set_ylim(-200, 200)
    ax.set_xlim(0, len(f_vec)) #make sure our x axis matched our chunk size
    line, = ax.plot(f_vec, list(map(int,fft_data)))
    line.set_ydata(list(map(int,fft_data)))
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.01)'''
    
    #plt.style.use('classic')
    #plt.rcParams['font.size']=18
    #fig = plt.figure(figsize=(13,8))
    #ax = fig.add_subplot(111)
    #plt.plot(fft_data)
    #ax.set_ylim([0,2*np.max(fft_data)])
    #plt.xlabel('Frequency [Hz]')
    #plt.ylabel('Amplitude [Pa]')
    #ax.set_xscale('log')
    #plt.grid(True)
    
    # max frequency resolution 
    #plt.annotate(r'$\Delta f_{max}$: %2.1f Hz' % (RATE/(2*CHUNK)),xy=(0.7,0.92),\
    #             xycoords='figure fraction')
    
    # annotate peak frequency
    #annot = ax.annotate('Freq: %2.1f'%(f_vec[max_loc]),xy=(f_vec[max_loc],fft_data[max_loc]),\
    #                    xycoords='data',xytext=(0,30),textcoords='offset points',\
    #                    arrowprops=dict(arrowstyle="->"),ha='center',va='bottom')
        
    #plt.savefig('fft_1kHz_signal.png',dpi=300,facecolor='#FCFCFC')
    plt.show()
