import numpy as np
import matplotlib.pyplot as plt
import wave
import socket
import time

# from scipy.fftpack import fft


RATE = 48000
RECORD_SEC = 5
WAVE_OUTPUT_FILENAME_0  = "samples\\hello\\speech_p30_1.wav"
SAMPLE_WIDTH = 2 #BYTES
CHUNK = 16384
num_elements = 2
HOST = '192.168.4.1'  # dia chi IP cua may tinh
PORT0 = 80         # cong ket noi

# Tao socket server
server_socket0 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

server_address_0 = (HOST, PORT0)
print("connecting to ports ")
print(WAVE_OUTPUT_FILENAME_0)
#print('Connected by', addr)
command = "record"
run = input()
time.sleep(4)
frames = []
record_data0 = []
server_socket0.connect(server_address_0)

#Receive data
def recv_data(desire_len):
# Nhận dữ liệu cho đến khi nhận đủ kích thước đã xác định
    data = b''
    while len(data) < desire_len:
        packet = server_socket0.recv(desire_len - len(data))
        if not packet:
            break
        data += packet
        
    return data

#Ham ghi am 
def creat_file(filename):
    wf = wave.open(filename, 'wb')
    wf.setnchannels(2)
    wf.setsampwidth(SAMPLE_WIDTH)
    wf.setframerate(RATE)
    return wf
        
#Ham ve do thi dang song      
#Vong lap thuc hien lenh

if command == "record":
    print (time.asctime( time.localtime(time.time()) ))
    wf0 = creat_file(WAVE_OUTPUT_FILENAME_0)
    data_len = 0
    for i in range (0, int(num_elements * RATE / CHUNK * RECORD_SEC * SAMPLE_WIDTH)):
        data0 = recv_data(CHUNK)
        #print(data0)
        data_len = data_len + len(data0)
        record_data0.append(data0)
        print (i)

    # print(data_len)
    wf0.writeframes(b''.join(record_data0 ))
    wf0.close()
    print (time.asctime( time.localtime(time.time()) ))


elif command == "print":
    while True:
        data = server_socket0.recv(CHUNK)
        record_data0.append(data)
        print(str(len(data)) + " " + str(len(record_data0)))
        print()
else: print("Invalid command")

arr_size = 4
with wave.open(WAVE_OUTPUT_FILENAME_0, 'rb') as wav_file:
    # Lấy thông tin về file WAV
    frame_rate = wav_file.getframerate()
    num_frames = wav_file.getnframes()
    num_channels = wav_file.getnchannels()

    # Đọc dữ liệu từ các loa và chuyển đổi thành mảng NumPy
    wav_data = wav_file.readframes(-1)
    wav_array = np.frombuffer(wav_data, dtype=np.int16)
    chan_0 = wav_array[0::4]
    chan_1 = wav_array[1::4]
    chan_2 = wav_array[2::4]
    chan_3 = wav_array[3::4]
# plt.subplot(121)
plt.plot(chan_0)
plt.plot(chan_1)
plt.plot(chan_2)
plt.plot(chan_3)
plt.legend(['chan0', 'chan1', 'chan2', 'chan3'])
plt.grid()

# freq = np.fft.rfftfreq(len(chan_0), 1/frame_rate)
# chan_0 = np.fft.rfft(chan_0)
# chan_1 = np.fft.rfft(chan_1)
# chan_2 = np.fft.rfft(chan_2)
# plt.subplot(122)
# plt.plot(freq, chan_0)
# plt.plot(freq, chan_1)
# plt.plot(freq, chan_2)
# plt.legend(['chan0', 'chan1', 'chan2'])
# plt.grid()

plt.show()