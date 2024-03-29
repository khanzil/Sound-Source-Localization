#include <driver/i2s.h>
#include <WiFi.h>
#include <WiFiClient.h>
#include <WiFiAP.h>

#define SERVER_PORT_0     80

#define I2S_WS_0 32
#define I2S_SD_0  35     
#define I2S_SD_1  14                                                               
#define I2S_SCK_0 33
#define I2S_SCK_1 27
#define I2S_WS_1 26

#define SAMPLE_RATE 48000
#define bufferLen   1024

WiFiServer server_0(SERVER_PORT_0);
WiFiClient client0;

const char* ssid = "Kien123";
const char* password = "abcde123";

int16_t sBuffer0[bufferLen];
int16_t sBuffer1[bufferLen];

void waitForWiFiConnectOrReboot(bool printOnSerial=true) {
  uint32_t notConnectedCounter = 0;
  while (WiFi.status() != WL_CONNECTED) {
      delay(100);
      if(printOnSerial) {
        Serial.println("Wifi connecting...");
      }
      notConnectedCounter++;
      if(notConnectedCounter > 50) { // Reset board if not connected after 5s
          if(printOnSerial) {
            Serial.println("Resetting due to Wifi not connecting...");
          }
          ESP.restart();
      }
  }
}

void i2s_install() {
  const i2s_config_t i2s_config = {
    .mode = i2s_mode_t(I2S_MODE_SLAVE | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = i2s_bits_per_sample_t(16),
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = i2s_comm_format_t(0X01),
    .intr_alloc_flags = 0, // default interrupt priority
    .dma_buf_count = 2,
    .dma_buf_len = bufferLen,
    .use_apll = false
  };

  i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);

  const i2s_config_t i2s_config_1 = {
    .mode = i2s_mode_t(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = i2s_bits_per_sample_t(16),
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = i2s_comm_format_t(0X01),
    .intr_alloc_flags = 0, // default interrupt priority
    .dma_buf_count = 2,
    .dma_buf_len = bufferLen,
    .use_apll = false
  };

  i2s_driver_install(I2S_NUM_1, &i2s_config_1, 0, NULL);
}

void i2s_0_setpin() {
  const i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_SCK_0,
    .ws_io_num = I2S_WS_0,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = I2S_SD_0
  };

  i2s_set_pin(I2S_NUM_0, &pin_config);
}

void i2s_1_setpin() {
  const i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_SCK_1,
    .ws_io_num = I2S_WS_1,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = I2S_SD_1
  };

  i2s_set_pin(I2S_NUM_1, &pin_config);
}

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  Serial.println("Setup I2S ...");
  
  if (!WiFi.softAP(ssid, password)) {
    log_e("Soft AP creation failed.");
    while(1);
  }
  IPAddress myIP = WiFi.softAPIP();
  Serial.print("AP IP address: ");
  Serial.println(myIP);
  // WiFi.begin(ssid, password);
  // waitForWiFiConnectOrReboot();

  delay(20);

  // Serial.println("Connected to the WiFi network");
  // Serial.println(WiFi.localIP());
  server_0.begin();
  // Serial.println("Socket server started on port " + String(SERVER_PORT_0));

  i2s_install();
  i2s_0_setpin();
  i2s_1_setpin();

  i2s_stop(I2S_NUM_0);
  i2s_stop(I2S_NUM_1);

  delay(5);

  i2s_start(I2S_NUM_0);
  i2s_start(I2S_NUM_1);
  
}

void loop() {
  size_t bytesIn0 = 0;
  size_t bytesIn1 = 0;
  int i = 0;
  
  client0 = server_0.available();
  
  if (client0.connected()) {
    delay(64);
  
    Serial.println("Clients connected");

    while(1){
      esp_err_t result0 = i2s_read(I2S_NUM_0, &sBuffer0, bufferLen*2, &bytesIn0, portMAX_DELAY);
      esp_err_t result1 = i2s_read(I2S_NUM_1, &sBuffer1, bufferLen*2, &bytesIn1, portMAX_DELAY);
      uint8_t data_char_0[bufferLen *4];
      for(i = 0; i < bufferLen; i++){
        // left i2s 0 is mic 1
        data_char_0[4*i + 1] = (uint8_t)((sBuffer0[i] >> 8) & 0xff);
        data_char_0[4*i + 0] = (uint8_t)((sBuffer0[i]) & 0xff);

        data_char_0[4*i + 3] = (uint8_t)((sBuffer1[i] >> 8) & 0xff);
        data_char_0[4*i + 2] = (uint8_t)((sBuffer1[i]) & 0xff);
      }
      client0.write((uint8_t*)&data_char_0,(int)(bufferLen * 4));
      if(!client0.connected()){
        Serial.println("Disconnected");
        break;
      }
    }
  }
  client0.stop();
}

