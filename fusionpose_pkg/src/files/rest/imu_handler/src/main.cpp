#include <Adafruit_SPIFlash.h>
#include <LSM6DS3.h>
#include <Wire.h>
#include <bluefruit.h>
#include <nrf_power.h>

// =======================
// Pins & LED definitions
// =======================
#define LEDR LED_RED
#define LEDG LED_GREEN
#define LEDB LED_BLUE

#define VBAT_ENABLE 14
#define VREF 3.7
#define ADC_MAX 4096

float getBatteryVoltage();

float getBatteryVoltage() {
  unsigned int adcCount = analogRead(PIN_VBAT);
  float adcVoltage = adcCount * VREF / ADC_MAX;
  return adcVoltage * 1510 / 510;
}


struct IMUDataPacket {
  // also include time
  uint32_t time;
  int16_t accel[3];
  int16_t gyro[3];
};

struct BatteryDataPacket {
  uint32_t time;
  float soc;
};

LSM6DS3 imu(I2C_MODE, 0x6A); //I2C device address 0x6A

// =======================
// SOC table
// =======================
const float voltages[] = {4.20, 4.15, 4.11, 4.08, 4.02, 3.98, 3.95, 3.91, 3.87, 3.85, 3.84, 3.82, 3.80, 3.79, 3.77, 3.75, 3.73, 3.71, 3.69, 3.61, 3.27};
const int   soc_values[] = {100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 0};
const int   num_entries = sizeof(voltages) / sizeof(voltages[0]);

float alpha = 0.001;  // Smoothing factor for low-pass filter
float smoothed_soc = 0;  // Smoothed SOC value
bool first_reading = true;  // Flag to track if this is the first SoC reading

// =======================
// IMU / DAQ configuration
// =======================
const float    freq = 100;         // Hz
const uint16_t accelRange = 4;     // g: Can be: 2, 4, 8, 16
const uint16_t gyroRange = 2000;   // dps: Can be: 125, 245, 500, 1000, 2000
const uint16_t accelSampleRate = 208; //Hz.  Can be: 13, 26, 52, 104, 208, 416, 833, 1666, 3332, 6664, 13330
const uint16_t gyroSampleRate = 208;  //Hz.  Can be: 13, 26, 52, 104, 208, 416, 833, 1666
const uint16_t accelBandWidth = 100;  //Hz.  Can be: 50, 100, 200, 400;
const uint16_t gyroBandWidth = 100;   //Hz.  Can be: 50, 100, 200, 400;

const uint32_t stayAwakeTimeMs = 1000*60; // 1 minute

#ifndef BLE_CONFIG_H
#define BLE_CONFIG_H

// =============================================
// === USER CONFIGURATION (ONLY CHANGE THIS!) ===
// =============================================
#define OID_NUMBER 25  // Change this number to update all identifiers

// =============================================
// === AUTOMATIC CONFIGURATION ===
// =============================================

// Convert number to string
#define STRINGIFY(x) #x
#define TO_STRING(x) STRINGIFY(x)

// Format two-digit OID string with leading zero if needed
#if (OID_NUMBER < 10)
  #define OID_STRING_UUID "0" TO_STRING(OID_NUMBER)
#else
  #define OID_STRING_UUID TO_STRING(OID_NUMBER)
#endif
#define OID_STRING TO_STRING(OID_NUMBER)

// Device name: e.g. "oid_07"
constexpr char DEVICE_NAME[] = "oid_" OID_STRING;

// UUID Suffix: e.g. "0700"
#define UUID_SUFFIX OID_STRING_UUID "00"

// BLE Service and Characteristic UUIDs
constexpr char BLE_SERVICE_UUID[]       = "19B10000-E8F2-537E-4F6C-D104768A" UUID_SUFFIX;
constexpr char BLE_IMU_CHAR_UUID[]      = "19B10001-E8F2-537E-4F6C-D104768A" UUID_SUFFIX;
constexpr char BLE_BATTERY_CHAR_UUID[]  = "19B10004-E8F2-537E-4F6C-D104768A" UUID_SUFFIX;

BLEService       bleService(BLE_SERVICE_UUID);
BLECharacteristic imuCharacteristic(BLE_IMU_CHAR_UUID);
BLECharacteristic batCharacteristic(BLE_BATTERY_CHAR_UUID);

Adafruit_FlashTransport_QSPI flashTransport;

// =======================
// Battery SoC helpers
// =======================
float interpolateSOC(float voltage) {
  // If voltage is higher than the max, return the highest SoC
  if (voltage >= voltages[0]) {
    return soc_values[0];
  }
  // If voltage is lower than the minimum, return the lowest SoC
  if (voltage <= voltages[num_entries - 1]) {
    return soc_values[num_entries - 1];
  }
  // Find the two surrounding voltage values and linearly interpolate
  for (int i = 0; i < num_entries - 1; i++) {
    if (voltage <= voltages[i] && voltage >= voltages[i + 1]) {
      // Perform linear interpolation
      float voltage_diff = voltages[i] - voltages[i + 1];
      float soc_diff = soc_values[i] - soc_values[i + 1];
      float voltage_ratio = (voltage - voltages[i + 1]) / voltage_diff;
      return soc_values[i + 1] + voltage_ratio * soc_diff;
    }
  }
  // Default return (shouldn't reach here)
  return -1;
}

float getCurrSOC() {
  // every 5 seconds, check battery voltage and update soc
  static unsigned long lastBatteryCheck = -5000;
  static float soc = 0;
  if (millis() - lastBatteryCheck > 5000) {
    lastBatteryCheck = millis();
    soc = interpolateSOC(getBatteryVoltage());
  }
  return soc;
}

// Function to apply low-pass filter
float lowPassFilter(float new_soc) {
  // If this is the first reading, initialize smoothed_soc with the new_soc
  if (first_reading) {
    smoothed_soc = new_soc;
    first_reading = false;
  } else {
    // Apply exponential smoothing for subsequent readings
    smoothed_soc = alpha * new_soc + (1 - alpha) * smoothed_soc;
  }
  return smoothed_soc;
}

// Disable QSPI flash to save power
void QSPIF_sleep(void)
{
  flashTransport.begin();
  flashTransport.runCommand(0xB9);
  flashTransport.end();
}

// =======================
// Wake on double tap
// =======================
void imuISR() {
  // Interrupt triggers for both single and double taps, so we need to check which one it was.
  uint8_t tapSrc;
  imu.readRegister(&tapSrc, LSM6DS3_ACC_GYRO_TAP_SRC);
  bool wasDoubleTap = (tapSrc & LSM6DS3_ACC_GYRO_DOUBLE_TAP_EV_STATUS_DETECTED) > 0;
  if (!wasDoubleTap) {
    nrf_power_system_off(NRF_POWER);
  }
}

void setupWakeUpInterrupt()
{
  // Tap interrupt code is based on code by daCoder
  // https://forum.seeedstudio.com/t/xiao-sense-accelerometer-examples-and-low-power/270801
  imu.settings.gyroEnabled = 0;
  imu.settings.accelEnabled = 1;
  imu.settings.accelSampleRate = 104;
  imu.settings.accelRange = 2;
  imu.begin();

  //https://www.st.com/resource/en/datasheet/lsm6ds3tr-c.pdf
  imu.writeRegister(LSM6DS3_ACC_GYRO_TAP_CFG1, 0b10001000); // Enable interrupts and tap detection on X-axis
  imu.writeRegister(LSM6DS3_ACC_GYRO_TAP_THS_6D, 0b10001000); // Set tap threshold
  const int duration = 0b0010 << 4; // 1LSB corresponds to 32*ODR_XL time
  const int quietTime = 0b10 << 2; // 1LSB corresponds to 4*ODR_XL time
  const int shockTime = 0b01 << 0; // 1LSB corresponds to 8*ODR_XL time
  imu.writeRegister(LSM6DS3_ACC_GYRO_INT_DUR2, duration | quietTime | shockTime); // Set Duration, Quiet and Shock time windows
  imu.writeRegister(LSM6DS3_ACC_GYRO_WAKE_UP_THS, 0x80); // Single & double-tap enabled (SINGLE_DOUBLE_TAP = 1)
  imu.writeRegister(LSM6DS3_ACC_GYRO_MD1_CFG, 0x08); // Double-tap interrupt driven to INT1 pin
  imu.writeRegister(LSM6DS3_ACC_GYRO_CTRL6_G, 0x10); // High-performance operating mode disabled for accelerometer

  // Set up the sense mechanism to generate the DETECT signal to wake from system_off
  pinMode(PIN_LSM6DS3TR_C_INT1, INPUT_PULLDOWN_SENSE);
  attachInterrupt(digitalPinToInterrupt(PIN_LSM6DS3TR_C_INT1), imuISR, CHANGE);

  return;
}

// =======================
// BLE advertising helpers
// =======================
void startAdvertising()
{
  Bluefruit.Advertising.addFlags(BLE_GAP_ADV_FLAGS_LE_ONLY_GENERAL_DISC_MODE);
  Bluefruit.Advertising.addTxPower();
  Bluefruit.Advertising.addService(bleService);

  Bluefruit.ScanResponse.addName();
  
  Bluefruit.Advertising.restartOnDisconnect(true);
  Bluefruit.Advertising.setInterval(128, 488);    // in unit of 0.625 ms
  Bluefruit.Advertising.setFastTimeout(30);      // number of seconds in fast mode
  Bluefruit.Advertising.start(0);                // 0 = Don't stop advertising after n seconds
}

void sleepUntilDoubleTap() {
  digitalWrite(LEDR, HIGH);
  digitalWrite(LEDG, HIGH);
  digitalWrite(LEDB, HIGH);

  Serial.println("Setting up interrupt");
  // Setup up double tap interrupt to wake back up
  setupWakeUpInterrupt();

  Serial.println("Entering sleep");
  Serial.flush();

  // Execution should not go beyond this
  nrf_power_system_off(NRF_POWER);
}
#endif

// =======================
// IMU configuration
// =======================
void setupImu() {
  imu.settings.accelRange = accelRange;
  imu.settings.accelSampleRate = accelSampleRate;
  imu.settings.accelBandWidth = accelBandWidth;
  imu.settings.gyroRange = gyroRange;
  imu.settings.gyroSampleRate = gyroSampleRate;
  imu.settings.gyroBandWidth = gyroBandWidth;

  imu.settings.accelODROff = 1;

  imu.begin();

  // // write high performance mode to on
  // imu.writeRegister(LSM6DS3_ACC_GYRO_CTRL6_G, 0x00);
}

// =======================
// Debug helpers
// =======================
void print_frequency(int* t, int* count) {
  int t_now = millis();
  if (t_now - *t > 2000) {
    // print t_now, t, count
    Serial.print("Time: ");
    Serial.print(t_now);
    Serial.print(", ");
    Serial.print("Last time: ");
    Serial.print(*t);
    Serial.print(", ");
    Serial.print("Frequency: ");
    Serial.print(float(*count)/float(t_now - *t)*1000);
    Serial.println(" Hz");
    *t = t_now;
    *count = 0;
  }
  *count = *count + 1;
}

// =======================
// DAQ / publishers
// =======================
void daq_publisher() {
  if (Serial) {
    static int t = millis();
    static int count = 0;
    print_frequency(&t, &count);
  }
  IMUDataPacket packet;
  packet.time = millis();
  packet.accel[0] = imu.readRawAccelX();
  packet.accel[1] = imu.readRawAccelY();
  packet.accel[2] = imu.readRawAccelZ();
  packet.gyro[0] = imu.readRawGyroX();
  packet.gyro[1] = imu.readRawGyroY();
  packet.gyro[2] = imu.readRawGyroZ();
  imuCharacteristic.notify(&packet, sizeof(packet));
  // every five seconds, print battery voltage
  static unsigned long lastBatteryCheck = 0;
  if (millis() - lastBatteryCheck > 5000) {
    lastBatteryCheck = millis();
    BatteryDataPacket batteryPacket;
    batteryPacket.time = millis();
    batteryPacket.soc = lowPassFilter(getCurrSOC());
    batCharacteristic.notify(&batteryPacket, sizeof(batteryPacket));
  }
}

// =======================
// Setup / loop
// =======================
void setup() {
  Serial.begin(9600);
  while (!Serial && millis() < 1000); // Timeout in case serial disconnected.

  analogReadResolution(ADC_RESOLUTION);
  pinMode(PIN_VBAT, INPUT);
  pinMode(VBAT_ENABLE, OUTPUT);
  digitalWrite(VBAT_ENABLE, LOW);

  pinMode(LEDR, OUTPUT);
  pinMode(LEDG, OUTPUT);
  pinMode(LEDB, OUTPUT);

  digitalWrite(LEDR, LOW); // red light before setup
  digitalWrite(LEDG, HIGH);
  digitalWrite(LEDB, HIGH);

  QSPIF_sleep();
  // report name and UUIDs
  Serial.println("This is IMU device: " + String(DEVICE_NAME));
  Serial.print("Service UUID: ");
  Serial.println(BLE_SERVICE_UUID);
  Serial.print("IMU Char UUID: ");
  Serial.println(BLE_IMU_CHAR_UUID);
  Serial.print("Battery Char UUID: ");
  Serial.println(BLE_BATTERY_CHAR_UUID);

  Bluefruit.autoConnLed(false);
  Serial.println("Initialize the Bluefruit nRF52 module");
  Bluefruit.configPrphBandwidth(4); // max is BANDWIDTH_MAX = 4
  Serial.print("Begin Bluefruit: ");
  Serial.println(Bluefruit.begin(1, 0));
  Bluefruit.Periph.setConnInterval(6, 6);
  Bluefruit.setName(DEVICE_NAME);
  Serial.println("Begin bleService");
  bleService.begin();

  // initialize imuCharacteristic
  imuCharacteristic.setProperties(CHR_PROPS_READ | CHR_PROPS_NOTIFY);
  imuCharacteristic.setPermission(SECMODE_OPEN, SECMODE_NO_ACCESS);
  imuCharacteristic.setFixedLen(sizeof(IMUDataPacket));
  Serial.println("Begin imuCharacteristic");
  imuCharacteristic.begin();
  IMUDataPacket initialPacket = { 0 };
  imuCharacteristic.write(&initialPacket, sizeof(initialPacket));

  // initialize batCharacteristic
  batCharacteristic.setProperties(CHR_PROPS_READ | CHR_PROPS_NOTIFY);
  batCharacteristic.setPermission(SECMODE_OPEN, SECMODE_NO_ACCESS);
  batCharacteristic.setFixedLen(sizeof(BatteryDataPacket));
  Serial.println("Begin batCharacteristic");
  batCharacteristic.begin();
  BatteryDataPacket initialBatPacket = { 0 };
  batCharacteristic.write(&initialBatPacket, sizeof(initialBatPacket));

  Serial.print("Starting IMU...");
  setupImu();
  Serial.println("Setup finished");
}

void print_info_periodically(unsigned int interval = 10000) {
    static unsigned long last_time = 0;
    if (millis() - last_time > interval) {
      // print device name
      Serial.print("Device name: ");
      Serial.println(DEVICE_NAME);
      Serial.print("Service UUID: ");
      Serial.println(BLE_SERVICE_UUID);
      Serial.print("IMU Char UUID: ");
      Serial.println(BLE_IMU_CHAR_UUID);
      Serial.print("Battery Char UUID: ");
      Serial.println(BLE_BATTERY_CHAR_UUID);
      last_time = millis();
    }
}

// create a function that blinks soc%*duration seconds blue and (100-soc)% red without blocking
void blinkBattery(int soc, int duration, int ledOn) {
  static int onDuration = soc/100.0*duration;
  static int redDuration = (100-soc)/100.0*duration;
  static uint32_t lastSwitch = millis();
  static bool isOn = true;
  if (millis() - lastSwitch > (isOn ? onDuration : redDuration)) {
    lastSwitch = millis();
    isOn = !isOn;
    digitalWrite(LEDR, isOn);
    digitalWrite(ledOn, !isOn);
    // update on duration when coming from red
    if (!isOn) {
      onDuration = soc/100.0*duration;
      redDuration = (100-soc)/100.0*duration;
    }
  }
}

void loop() {
  int start = millis();

  Serial.print("Starting advertising...");
  startAdvertising();
  digitalWrite(LEDR, HIGH);
  digitalWrite(LEDG, HIGH);
  digitalWrite(LEDB, LOW);

  unsigned long wakeUpTime = millis();

  while (millis() - wakeUpTime < stayAwakeTimeMs) {
    if (Bluefruit.connected(0)) {
      Serial.println("Connected");
      digitalWrite(LEDR, HIGH);
      digitalWrite(LEDG, LOW);
      digitalWrite(LEDB, HIGH); // green light when connected
      while (Bluefruit.connected(0)) {
        start = millis();
        daq_publisher();
        while (millis() - start < 1000/freq);
        blinkBattery(lowPassFilter(getCurrSOC()), 10000, LEDG);
        if (Serial) {
          print_info_periodically();
        }
      }
      digitalWrite(LEDR, HIGH);
      digitalWrite(LEDG, HIGH);
      digitalWrite(LEDB, LOW); // blue light when not connected
      // reset wakeUpTime to prevent immediate sleep
      wakeUpTime = millis();
    }
    // log soc
    int soc = getCurrSOC();
    blinkBattery(lowPassFilter(getCurrSOC()), 10000, LEDB);

    if (Serial) {
      wakeUpTime = millis();     // Don't sleep if USB connected, to make code upload easier.
      Serial.print("Battery voltage: ");
      Serial.print(getBatteryVoltage());
      Serial.print(", SoC: ");
      Serial.print(soc);
      Serial.print("%, smoothed SoC: ");
      Serial.print(lowPassFilter(soc));
      Serial.println("%");
      print_info_periodically();
    }
    delay(100);
  }
  Serial.println("Stopping advertising");
  Bluefruit.Advertising.stop();
  sleepUntilDoubleTap();
}
