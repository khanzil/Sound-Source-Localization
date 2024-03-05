from doa_func import *


with wave.open("samples\\hello\\speech_p40_2.wav", 'rb') as wav_file:
    # Lấy thông tin về file WAV
    frame_rate = wav_file.getframerate()
    num_frames = wav_file.getnframes()
    num_channels = wav_file.getnchannels()

    # Đọc dữ liệu từ các loa và chuyển đổi thành mảng NumPy
    wav_data = wav_file.readframes(-1)
    wav_array = np.frombuffer(wav_data, dtype=np.int16)
    chan_1 = wav_array[1::4]
    chan_2 = wav_array[3::4]
# plt.subplot(121)
    
out1 = ss.resample(chan_1, int(len(chan_1)/frame_rate * 1024e3))
out2 = ss.resample(chan_2, int(len(chan_2)/frame_rate * 1024e3))

plt.plot(out1)
plt.plot(out2)
plt.grid()

plt.show()