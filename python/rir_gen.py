import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import rir_generator
from doa_func import *

file_name = 'samples\\hello\\hello.wav'

with wave.open(file_name, 'rb') as wav_file:
    # Lấy thông tin về file WAV
    frame_rate = wav_file.getframerate()
    num_frames = wav_file.getnframes()
    num_channels = wav_file.getnchannels()

    # Đọc dữ liệu từ các loa và chuyển đổi thành mảng NumPy
    wav_data = wav_file.readframes(-1)
    wav_array = np.frombuffer(wav_data, dtype=np.int16)[::2] * 2

array = np.copy(wav_array)

angle = 40*np.pi/180
# pad = np.zeros(int(frame_rate * T60))
# array = np.append(pad, array)
rir = rir(angle)

out1 = np.array([])
out2 = np.array([])
out3 = np.array([])

new_array = ss.resample(array, int(len(array)*1024e3/frame_rate))

out1 = np.convolve(new_array, rir[:,0], mode = 'full')
out2 = np.convolve(new_array, rir[:,1], mode = 'full')
out3 = np.convolve(new_array, rir[:,2], mode = 'full')
# out1 += 20*np.random.normal(0,1,len(out1))
# out2 += 20*np.random.normal(0,1,len(out2))

plt.subplot(211)
plt.plot(out1)
plt.plot(out2)
plt.plot(out3)
plt.grid()

out1 = np.array(ss.resample(out1, int(len(out1)/(1024e3)*frame_rate)))
out2 = np.array(ss.resample(out2, int(len(out2)/(1024e3)*frame_rate)))
out3 = np.array(ss.resample(out3, int(len(out3)/(1024e3)*frame_rate)))

plt.subplot(212)
plt.plot(out1)
plt.plot(out2)
plt.plot(out3)
plt.grid()
plt.show()

out1 = out1.astype(np.int16)
out2 = out2.astype(np.int16)
out3 = out3.astype(np.int16)
newfile = wave.open("samples\\rir_gen_3\\hello_rir_1_p40.wav", 'wb')
newfile.setframerate(frame_rate)
newfile.setnchannels(1)
newfile.setsampwidth(2)
newfile.setnframes(len(out1))
newfile.writeframes(out1.tobytes())

newfile = wave.open("samples\\rir_gen_3\\hello_rir_2_p40.wav", 'wb')
newfile.setframerate(frame_rate)
newfile.setnchannels(1)
newfile.setsampwidth(2)
newfile.setnframes(len(out1))
newfile.writeframes(out2.tobytes())


newfile = wave.open("samples\\rir_gen_3\\hello_rir_3_p40.wav", 'wb')
newfile.setframerate(frame_rate)
newfile.setnchannels(1)
newfile.setsampwidth(2)
newfile.setnframes(len(out1))
newfile.writeframes(out3.tobytes())




