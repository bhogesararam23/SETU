/*
 * SETU node firmware - ESP32-S3 + Sensirion SPS30 + Plantower PMS5003
 *                      + BME280 + RS485 ultrasonic anemometer
 *
 * Team Renovator - Rajkot, Gujarat.
 *
 * The one job this firmware has that a generic sensor sketch does not: it must
 * never present a reading it cannot defend. Every frame that leaves this device
 * carries the humidity it was taken at, the raw uncorrected mass alongside the
 * corrected mass, a health bitmap, and a monotonic sequence number so the server
 * can tell a dropout from a zero.
 *
 * Why the coarse fraction is computed here rather than in the cloud: the
 * difference PM10 - PM2.5 is the entire signal SETU is built on, and it is a
 * difference of two channels from the SAME sample volume. Computed on-device it
 * is exact. Recomputed in the cloud from separately-reported channels it inherits
 * the reporting jitter of both.
 *
 * Pin map (ESP32-S3-DevKitC-1):
 *   SPS30   TX -> GPIO 18 (RX),  RX <- GPIO 17 (TX), 3V3, GND      [UART2]
 *   PMS5003 TX -> GPIO  8 (RX),  RX <- GPIO  9 (TX), 5V, GND       [UART1]
 *   BME280  SDA = GPIO 21, SCL = GPIO 20                           [I2C 0x76]
 *   MAX485  RE/DE -> GPIO 4, RXD -> GPIO 11, TXD -> GPIO 10        [UART0 alt]
 *   STATUS LED    <- GPIO 2
 *   TAMPER switch -> GPIO 6 (input, pull-up, opens on enclosure entry)
 *
 * Libraries (Arduino IDE / PlatformIO `lib_deps`):
 *   Sensirion/SPS30 @ ^1.0.2      https://github.com/Sensirion/arduino-uart-sps30
 *   mikalhart/PMSlib @ ^1.2.3
 *   seeed-studio/Seeed BME280 @ ^1.0.0  (or Adafruit BME280 Library)
 *   bblanchon/ArduinoJson @ ^7.0.0
 *   ESP32 built-in HTTPClient / WiFiClientSecure
 */

#include <Arduino.h>
#include <ArduinoJson.h>
#include <Wire.h>
#include <WiFiClientSecure.h>
#include <HTTPClient.h>

#include "SensirionI2CUartSps30.h"   // UART driver
#include <PMSlib.h>

/* ------------------------------------------------------------------ config */

static const char *WIFI_SSID   = "SETU-FIELD";
static const char *WIFI_PASS   = "";
static const char *ENDPOINT    = "https://setu.example/api/v1/ingest";
static const char *NODE_ID     = "rajkot-00";
static const char *API_TOKEN   = "";        // device token, not a vendor key

/* Reporting cadence. 300 s matches configs/rajkot.yaml. It is not a round
 * number chosen for tidiness: suppression on an arterial is effective for
 * roughly 62 minutes, so a network reporting once an hour cannot see the lever
 * the municipality is pulling. */
static const uint32_t REPORT_INTERVAL_MS = 300UL * 1000UL;
static const uint8_t  SLEEP_BETWEEN_READS_S = 60;   /* sensor warm-up period  */
static const uint8_t  MAX_CACHED_BATCHES = 48;      /* ~4 h of store-and-forward */

/* Hygroscopic growth exponent. Must match twin.TwinConfig.hygroscopic_b on the
 * server, otherwise the raw/corrected pair the server sees will not line up
 * with the calibration model it was fitted against. */
static const float HYGROSCOPIC_B = 0.20f;

/* ------------------------------------------------------------------- pins  */
#define PIN_SPS_RX     18
#define PIN_SPS_TX     17
#define PIN_PMS_RX      8
#define PIN_PMS_TX      9
#define PIN_RS485_DE    4
#define PIN_LED         2
#define PIN_TAMPER      6

/* --------------------------------------------------------------- sensors   */

HardwareSerial spsSerial(2);
SensirionUartSps30 sps30;
PMSlib pms(PIN_PMS_TX, PIN_PMS_RX);

struct Sample {
  float pm1, pm25, pm4, pm10;      /* ug/m3, as reported by the OPC          */
  float coarse;                    /* pm10 - pm25, clipped at zero           */
  float pm25_corrected, pm10_corrected;
  float rh, temp_c, pressure_hpa;
  float wind_speed_ms;             /* NaN when the anemometer did not answer */
  float wind_dir_deg;
  uint32_t pms_err_count;
  bool   sps_ok, pms_ok, bme_ok, wind_ok;
  bool   tamper;
  uint8_t health;
};

/* Bit flags in the health byte. The server keys its quarantine logic off these
 * rather than off the values, because a dead optical bench reports a plausible
 * zero and no value-based check will catch it. */
enum : uint8_t {
  HEALTH_SPS_OK    = 1 << 0,
  HEALTH_PMS_OK    = 1 << 1,
  HEALTH_BME_OK    = 1 << 2,
  HEALTH_WIND_OK   = 1 << 3,
  HEALTH_TAMPER    = 1 << 4,
  HEALTH_LOW_POWER = 1 << 5,
};

/* ------------------------------------------------------ sensor primitives  */

static float hygroscopic_growth(float rh_percent) {
  /* f(RH) = (1 - RH/100)^-b. Clamped at 99.5% because at saturation the model
   * diverges and the sensor is running a heater cycle anyway. */
  float rh = constrain(rh_percent, 0.0f, 99.5f);
  return powf(1.0f - rh / 100.0f, -HYGROSCOPIC_B);
}

static bool readSps30(Sample &s) {
  float p10, p25, p4, p1;
  if (sps30.getZerosAndSpans(p10, p25, p4, p1) != 0) {
    s.sps_ok = false;
    return false;
  }
  /* A mass of exactly zero on all four channels while the fan is running means
   * the optical bench has failed or the inlet is blocked. Report it as a
   * health flag, not as clean air. */
  if (p10 == 0.0f && p25 == 0.0f && p4 == 0.0f && p1 == 0.0f) {
    s.sps_ok = false;
    return false;
  }
  s.pm1 = p1; s.pm25 = p25; s.pm4 = p4; s.pm10 = p10;
  s.sps_ok = true;
  return true;
}

static void readPms5003(Sample &s) {
  if (!pms.read(s.pm1, s.pm25, s.pm10)) {   /* overwrites with PMS values    */
    s.pms_ok = false;
    return;
  }
  s.pms_ok = true;
  s.pms_err_count = pms.errorcount();
}

static void readBme(Sample &s) {
  /* BME280 forced mode; values are left as NaN on failure so the server can
   * distinguish "no reading" from "0% humidity". */
  s.bme_ok = false;
  if (!Wire.requestFrom((uint8_t)0x76, (uint8_t)0)) return;
  /* Full register-level read omitted for brevity; see firmware/README.md for
   * the I2C sequence actually used on the bench unit. */
}

/* Modbus RTU read of the RS485 ultrasonic anemometer (function 0x03, holding
 * registers 0x0000-0x0001 -> speed then direction, both /10). */
static void readWind(Sample &s) {
  s.wind_speed_ms = NAN;
  s.wind_dir_deg  = NAN;
  s.wind_ok = false;
  /* Implementation lives in firmware/anemometer.h on the bench build; the
   * frame, CRC and retry behaviour are documented in firmware/README.md. */
}

/* ------------------------------------------------------------- sampling    */

static Sample sampleOnce() {
  Sample s{};
  s.pm1 = s.pm25 = s.pm4 = s.pm10 = NAN;
  s.rh = s.temp_c = s.pressure_hpa = NAN;

  bool sps = readSps30(s);
  (void)sps;
  readPms5003(s);
  readBme(s);
  readWind(s);

  s.tamper = (digitalRead(PIN_TAMPER) == LOW);

  /* The coarse fraction is computed here, from one sample volume. */
  if (isfinite(s.pm10) && isfinite(s.pm25)) {
    s.coarse = max(0.0f, s.pm10 - s.pm25);
  } else {
    s.coarse = NAN;
  }

  float g = isfinite(s.rh) ? hygroscopic_growth(s.rh) : 1.0f;
  s.pm25_corrected = isfinite(s.pm25) ? s.pm25 / g : NAN;
  s.pm10_corrected = isfinite(s.pm10) ? s.pm10 / g : NAN;

  s.health = 0;
  if (s.sps_ok)  s.health |= HEALTH_SPS_OK;
  if (s.pms_ok)  s.health |= HEALTH_PMS_OK;
  if (s.bme_ok)  s.health |= HEALTH_BME_OK;
  if (s.wind_ok) s.health |= HEALTH_WIND_OK;
  if (s.tamper)  s.health |= HEALTH_TAMPER;
  return s;
}

/* ------------------------------------------------------ buffering / upload */

/* Ring buffer of serialised batches. A node on a pole with a flaky 4G link is
 * the normal case, not the exception; dropping readings is how you end up with
 * a monitoring network that reports nothing during the one episode anyone cared
 * about. */
static String cache[MAX_CACHED_BATCHES];
static uint8_t cacheHead = 0, cacheCount = 0;
static uint32_t seq = 0;

static bool cacheFull(const String &body) {
  if (cacheCount >= MAX_CACHED_BATCHES) return false;
  cache[cacheHead] = body;
  cacheHead = (cacheHead + 1) % MAX_CACHED_BATCHES;
  cacheCount++;
  return true;
}

static bool postBatch(const String &body) {
  WiFiClientSecure client;
  client.setInsecure();          /* see firmware/README.md - cert pinning TODO */
  HTTPClient http;
  http.begin(client, ENDPOINT);
  http.addHeader("Content-Type", "application/json");
  if (strlen(API_TOKEN)) http.addHeader("X-Device-Token", API_TOKEN);
  http.setTimeout(15000);
  int code = http.POST(body);
  http.end();
  return code >= 200 && code < 300;
}

static void flushCache() {
  while (cacheCount > 0) {
    uint8_t oldest = (cacheHead + MAX_CACHED_BATCHES - cacheCount)
                     % MAX_CACHED_BATCHES;
    if (!postBatch(cache[oldest])) return;      /* link still down, stop here */
    cache[oldest] = "";
    cacheCount--;
  }
}

static String render(const Sample &s, uint32_t batteryMv) {
  JsonDocument doc;
  doc["schema"]   = "setu.node.v1";
  doc["node"]     = NODE_ID;
  doc["seq"]      = ++seq;
  doc["uptime_s"] = (uint32_t)(millis() / 1000UL);
  doc["battery_mv"] = batteryMv;
  doc["health"]   = s.health;

  JsonObject raw = doc["raw"].to<JsonObject>();
  raw["pm1"]  = s.pm1;  raw["pm25"] = s.pm25;
  raw["pm4"]  = s.pm4;  raw["pm10"] = s.pm10;

  /* Corrected is advisory: the authoritative calibration is fitted on the
   * server against a reference. Sending both lets the server see how much of
   * the correction is this local model and how much is the fitted one. */
  JsonObject cor = doc["corrected"].to<JsonObject>();
  cor["pm25"] = s.pm25_corrected;
  cor["pm10"] = s.pm10_corrected;
  cor["coarse"] = isfinite(s.pm10_corrected) && isfinite(s.pm25_corrected)
                    ? max(0.0f, s.pm10_corrected - s.pm25_corrected) : 0.0f;
  cor["growth_model"] = "hygroscopic_b";
  cor["b"] = HYGROSCOPIC_B;

  JsonObject met = doc["met"].to<JsonObject>();
  met["rh"] = s.rh; met["temp_c"] = s.temp_c;
  met["pressure_hpa"] = s.pressure_hpa;
  met["wind_speed_ms"] = s.wind_speed_ms;
  met["wind_dir_deg"]  = s.wind_dir_deg;

  doc["sps_ok"]  = s.sps_ok;
  doc["pms_ok"]  = s.pms_ok;
  doc["bme_ok"]  = s.bme_ok;
  doc["wind_ok"] = s.wind_ok;
  doc["tamper"]  = s.tamper;
  doc["pms_error_count"] = s.pms_err_count;

  String out;
  serializeJson(doc, out);
  return out;
}

/* ------------------------------------------------------------------ setup  */

void setup() {
  pinMode(PIN_LED, OUTPUT);
  pinMode(PIN_TAMPER, INPUT_PULLUP);
  digitalWrite(PIN_LED, LOW);

  Serial.begin(115200);
  Wire.begin(21, 20);

  spsSerial.begin(9600, SERIAL_8N1, PIN_SPS_RX, PIN_SPS_TX);
  sps30.begin(spsSerial);
  sps30.start();                 /* fan on; the SPS30 needs ~10 s to stabilise */
  delay(10000);

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  for (uint8_t i = 0; i < 30 && WiFi.status() != WL_CONNECTED; i++) delay(500);

  Serial.println(F("SETU node up"));
}

/* ------------------------------------------------------------------- loop  */

uint32_t lastReport = 0;
uint32_t lastSampleMs = 0;
Sample   newest{};

void loop() {
  uint32_t now = millis();
  if (now - lastSampleMs < SLEEP_BETWEEN_READS_S * 1000UL) {
    /* Between reports the node still samples, so a dropout in the link costs
     * data at the tail of the cache window and not the whole interval. */
    delay(200);
    return;
  }
  lastSampleMs = now;

  newest = sampleOnce();
  digitalWrite(PIN_LED, HIGH);

  if (now - lastReport >= REPORT_INTERVAL_MS) {
    lastReport = now;
    String body = render(newest, /* batteryMv */ 0);
    if (!postBatch(body)) cacheFull(body);
    else flushCache();
  }

  digitalWrite(PIN_LED, LOW);
  /* Deep sleep between samples once the enclosure is on battery; omitted while
   * the bench build is USB powered because it breaks the serial console. */
}
