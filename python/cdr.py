from doa_func import *

doa = np.array([0])
doa12 = 0
doa13 = 0
doai = 0
doax = np.array([0])
doay = np.array([0])
file_name = 'samples\\rir_gen_3\\hello_rir_1_n150.wav'
with wave.open(file_name, 'rb') as wav_file:
    frame_rate = wav_file.getframerate()
    num_frames = wav_file.getnframes()
    num_channels = wav_file.getnchannels()

    wav_data = wav_file.readframes(-1)
    wav_array = np.frombuffer(wav_data, dtype=np.int16)
    channel_1 = wav_array /10

file_name = 'samples\\rir_gen_3\\hello_rir_2_n150.wav'
with wave.open(file_name, 'rb') as wav_file:
    wav_data = wav_file.readframes(-1)
    wav_array = np.frombuffer(wav_data, dtype=np.int16)
    channel_2 = wav_array /10

file_name = 'samples\\rir_gen_3\\hello_rir_3_n150.wav'
with wave.open(file_name, 'rb') as wav_file:
    wav_data = wav_file.readframes(-1)
    wav_array = np.frombuffer(wav_data, dtype=np.int16)
    channel_3 = wav_array /10

file_name = "samples\\hello\\speech_n20_1.wav"
with wave.open(file_name, 'rb') as wav_file:
    # Lấy thông tin về file WAV
    frame_rate = wav_file.getframerate()
    num_frames = wav_file.getnframes()
    num_channels = wav_file.getnchannels()
    samp = wav_file.getsampwidth()
    # Đọc dữ liệu từ các loa và chuyển đổi thành mảng NumPy
    wav_data = wav_file.readframes(-1)
    wav_array = np.frombuffer(wav_data, dtype=np.int16)
    channel_1 = wav_array[0::2]
    channel_2 = wav_array[1::2]

G_old12 = np.ones(3*(FRAME+1)).reshape(3, (FRAME+1))
G_old13 = np.ones(3*(FRAME+1)).reshape(3, (FRAME+1))

for k in range(int(len(channel_1)/FRAME)):
    data_1 = channel_1[k*FRAME:(k+1)*FRAME]
    data_2 = channel_2[k*FRAME:(k+1)*FRAME]
    data_3 = channel_3[k*FRAME:(k+1)*FRAME]
    channel_1_fft = np.fft.rfft(data_1, 2*FRAME)
    channel_2_fft = np.fft.rfft(data_2, 2*FRAME)
    channel_3_fft = np.fft.rfft(data_3, 2*FRAME)

    M12, G_old12 = cdr_cal(channel_1_fft, channel_2_fft, frame_rate, G_old12)
    M13, G_old13 = cdr_cal(channel_1_fft, channel_3_fft, frame_rate, G_old13)
    if(SNR(data_1) > 8):
        doa12, spec12 = GCCPHAT(channel_1_fft, channel_2_fft, frame_rate, M=M12)
        doa13, spec12 = GCCPHAT(channel_1_fft, channel_3_fft, frame_rate, M=M13)
        if (np.abs(np.sin(doa12) ** 2 + np.sin(doa13) ** 2 - 1) > 10**(-1)):
            doai = 0
        elif (doa13 < 0):
            doai = np.sign(doa12) * np.pi - doa12
        else:
            doai = doa12
    else:
        doai = 0
    print(np.round(doa12*180/np.pi*10)/10)
    doa = np.append(doa, doai)
    doax = np.append(doax, doa12)
    doay = np.append(doay, doa13)

doa = np.round(doa*180/np.pi*10)/10
doax = np.round(doax*180/np.pi*10)/10
doay = np.round(doay*180/np.pi*10)/10
# print(doa)
plt.subplot(211)
# plt.plot(doa)
plt.plot(doax)
# plt.plot(doay)

plt.legend(['DOA','Angle Ox', 'Angle Oy'])


plt.grid()

plt.subplot(212)
plt.plot(channel_1)
plt.plot(channel_2)
plt.grid()


plt.show()




