import wave
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
import rir_generator
import time
import socket

dis = 0.06 * 2
sound_velo = 340
num_sources = 1
num_elements = 4
radius = 1
FRAME = 1024
Noise = 43
gcc_reso = 100
cdr_thrs = 0.2
Ffactor = 0.8
T60 = 0.4

def num_sources_est(array):
    noise_T = np.max(array)*0.8
    signal = []
    signal = array[array> noise_T]
    return len(signal)

def pass_band(spectrum, fre_min, fre_max,fs):
    freqs = np.fft.fftfreq(len(spectrum), d= 1/fs)
    spectrum[(freqs < fre_min)] = 0
    spectrum[(freqs > fre_max)] = 0
    return spectrum

def lookupdelay(angles, freqs):
    array = np.linspace(1, num_elements-1, num_elements-1)
    timedelay = (1j * np.sin(angles) * 2 * np.pi * dis / sound_velo)
    table = np.exp(np.outer(array, (np.outer(timedelay, freqs))).reshape(num_elements-1, 361, len(freqs)))
    return table

def SNR(chan):
    a = np.linalg.norm(chan) ** 2 / len(chan)
    if (a-Noise <= 0 ):
        return -100
    return 10*np.log10((a-Noise)/Noise)

def srpphat(covmat):
    av = np.ones(num_elements)
    val = np.abs(av.transpose() @ covmat @ av)
    return val

def angle_shift(data, angle, idx_ele, frame_rate):
    fft_data = np.fft.fft(data)
    freq = np.fft.fftfreq(len(data), 1/frame_rate)
    shift_fft = fft_data * np.exp(-1j * 2 * np.pi * freq * idx_ele * dis * np.sin(angle)/sound_velo)
    return np.real(np.fft.ifft(shift_fft))

def music(CovMat,arr_size, Angles):
    # CovMat is the signal covariance matrix, L is the number of sources, N is the number of antennas
    # array holds the positions of antenna elements
    # Angles are the grid of directions in the azimuth angular domain
    eig_val,eig_vector = np.linalg.eig(CovMat)
    # tri rien va vector rieng can duoc xep tu lon den be 
    eig_val = np.abs(eig_val)
    idx_min = np.argsort(eig_val)[:(arr_size - num_sources_est(eig_val))]
    idx_min = np.flip(idx_min)
    Qn  = eig_vector[:,idx_min]
    av = np.array([1/radius, 
                   1/np.sqrt((radius * np.sin(Angles)+0.06)**2 + (radius * np.cos(Angles))**2 ), 
                   1/np.sqrt((radius * np.sin(Angles)+0.12)**2 + (radius * np.cos(Angles))**2 ),
                   1/np.sqrt((radius * np.sin(Angles)+0.18)**2 + (radius * np.cos(Angles))**2 ),
                   ]).T
    pspectrum = 1/((np.abs(av.transpose() @ Qn @ Qn.conj().transpose() @ av)))
    pspectrum = np.log(pspectrum)

    return pspectrum

def GCCPHAT(fft1, fft2, frame_rate, M = []):
    if M==[]:
        M = np.ones(len(fft1))
    n = 2 * gcc_reso * len(fft1)
    max_delay = int(np.floor(dis/sound_velo*frame_rate*gcc_reso))

    G12 = M * fft2 * fft1.conj()
    Gabs = np.abs(G12)
    Gabs[G12 == 0] = 1

    spec12 = np.abs(np.fft.irfft(G12/Gabs, n = n).real)
    # plt.plot(spec12)
    # plt.grid()
    # plt.show()

    spec12 = (np.append(spec12[-max_delay:], spec12[0:max_delay+1]))
    tdoa = int(np.argmax(spec12))
 
    # x = np.linspace(-max_delay, max_delay, 2*max_delay+1)
    # x = np.arcsin(x/(dis/sound_velo*frame_rate*gcc_reso))
    # x = np.round(x*180/np.pi*10)/10
    # plt.plot(x,spec12)
    # plt.plot(x[tdoa], spec12[tdoa])
    # plt.grid()
    # plt.show()

    doa = np.arcsin((tdoa-max_delay)/(dis/sound_velo*frame_rate*gcc_reso))
    doa = np.arctan((np.sin(doa)-0.06)/np.cos(doa))
    return doa, spec12

def cdr_cal(fft1, fft2, frame_rate, G_old):
    freqs = np.fft.rfftfreq(len(fft1)*2-1, d = 1/frame_rate)
    M = np.zeros(len(freqs))
    Gnoise = np.sinc(2 * freqs * dis / sound_velo)
    G11 = (1-Ffactor) * fft1 * fft1.conjugate() + Ffactor * G_old[0,:]
    G22 = (1-Ffactor) * fft2 * fft2.conjugate() + Ffactor * G_old[1,:]
    G12 = (1-Ffactor) * fft1 * fft2.conjugate() + Ffactor * G_old[2,:]

    G_old = np.asanyarray([G11, G22, G12])
    Gx = G12/np.sqrt(G11*G22+10**-9)
    G = np.sqrt(Gnoise**2 * (np.real(Gx)**2 - np.abs(Gx)**2 + 1) - 2 * Gnoise * np.real(Gx) + np.abs(Gx)**2)

    CDR = (Gnoise * np.real(Gx) - np.abs(Gx) ** 2 - G)/(np.abs(Gx)**2 - 1 +10**-9)
    M[1/(CDR+1) < cdr_thrs] = 1
    M[np.abs(freqs)<200] = 0
    return M, G_old

def rir(angle):
    rir = rir_generator.generate(
    c = sound_velo,
    fs = 1024e3,
    r = [
            [2, 1.5, 1],
            [2, 1.5-0.12, 1],
            [2-0.12, 1.5, 1],
        ],
    s = [2+np.cos(angle), 1.5+np.sin(angle), 1],
    L = [5, 4, 3],
    reverberation_time = T60,
    nsample = int(T60 * 1024e3),
    dim = 3,
    )
    # plt.plot(rir[:, 0])
    # plt.plot(rir[:, 1])
    # plt.grid()
    # plt.show()
    return rir

def recv_data(desire_len, server):
# Nhận dữ liệu cho đến khi nhận đủ kích thước đã xác định
    data = b''
    while len(data) < desire_len:
        packet = server.recv(desire_len - len(data))
        if not packet:
            break
        data += packet
        
    return data




