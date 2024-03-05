from doa_func import *


frame_rate =    44100
FRAME_SIZE =    8192
SAMP_WIDTH =    2   #bytes
CHUNK =         4*FRAME_SIZE #Bytes 4 channel 2 bytes

HOST = "192.168.4.1"       #dia chi IP cua esp32
PORT = 80
log_file_name = 'log.txt'

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

server_address = (HOST, PORT)
print("Connecting to port")
try:
    server_socket.connect(server_address)
    print("Connected! Ready to receive data.")
    doai = 0
    threshold = 15

    data = np.zeros(CHUNK * 2)
    G_old = np.zeros(3*(FRAME+1)).reshape(3, (FRAME+1)) + 1j*np.zeros(3*(FRAME+1)).reshape(3, (FRAME+1))

    dec_data = np.frombuffer(recv_data(CHUNK* SAMP_WIDTH, server_socket), dtype= np.int16)
    data[0: CHUNK] = dec_data

    while(True):
        start_time = time.time()

        frame_data = recv_data(CHUNK*SAMP_WIDTH, server_socket)
        data[CHUNK:] = np.frombuffer(frame_data, dtype= np.int16)
        data_1 = data[1::4]
        data_2 = data[3::4]
        data_1_fft = np.fft.rfft(data_1, 2*FRAME)
        data_2_fft = np.fft.rfft(data_2, 2*FRAME)
        M, G_old = cdr_cal(data_1_fft, data_2_fft, frame_rate, G_old)

        if(SNR(data_1) > 10):
            doai, spec12 = GCCPHAT(data_1_fft, data_2_fft, frame_rate, M=M)
            print(doai)       

        # ngat khoi vong lap va xoa hang doi khi thoi gian xu ly qua lau
        # loop_time = time.time() - start_time
        
        # if(loop_time > 0.4 and count_frame < 10):
        #     clear_data = recv_data((10 - count_frame)* CHUNK * SAMP_WIDTH, server_socket)
        #     data[CHUNK:] = np.frombuffer(clear_data, dtype= np.int16)[-CHUNK:]
        #     break
        data[:CHUNK] = data[CHUNK:]

except socket.error as err:
    print ("socket creation failed with error %s" %(err))        
    


