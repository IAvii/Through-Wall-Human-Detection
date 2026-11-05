#include <WiFi.h>
#include "esp_wifi.h"
#include "esp_wifi_types.h"
#include "nvs_flash.h"
#include "esp_event.h"
#include "esp_netif.h"
#include "ping/ping_sock.h"

/* ── User Configuration ─────────────────────────────────── */
#define WIFI_SSID       "AirFiber-Panchal"
#define WIFI_PASS       "Panchal@1477"
#define SERIAL_BAUD     2000000              // 2 Mbaud — must match config.py
#define PING_INTERVAL_MS  10                 // 100 Hz ping rate

/* ── GPIO Pin Assignments ───────────────────────────────── */
#define PIN_GREEN_1   D0  // D0 — Green LED 1
#define PIN_RED_1     D1   // D1 — Red LED 1
#define PIN_BUZZER    D2  // D2 — Buzzer via BC547
#define PIN_RED_2     D6  // D6 — Red LED 2
#define PIN_GREEN_2   D7  // D7 — Green LED 2

/* ── Buzzer State ───────────────────────────────────────── */
static unsigned long buzzer_start_time = 0;
static bool buzzer_active = false;

/* ── Volatile CSI data shared between callback and loop ── */
static volatile bool csi_data_ready = false;
static int8_t  csi_raw_buf[512];
static int     csi_raw_len   = 0;
static int     csi_rssi      = 0;
static unsigned long csi_timestamp = 0;

/* ── Debug counters ─────────────────────────────────────── */
static volatile uint32_t csi_callback_count = 0;
static uint32_t csi_sent_count = 0;

/* ═══════════════════════════════════════════════════════════
 * CSI CALLBACK — runs in WiFi task context, keep minimal
 * ═══════════════════════════════════════════════════════════ */
void IRAM_ATTR wifi_csi_callback(void *ctx, wifi_csi_info_t *info) {
    if (info == NULL || info->buf == NULL) return;
    if (info->len <= 0 || info->len > (int)sizeof(csi_raw_buf)) return;

    csi_callback_count++;

    // Drop if previous data hasn't been consumed yet
    if (csi_data_ready) return;

    // Copy data out of callback context
    memcpy(csi_raw_buf, info->buf, info->len);
    csi_raw_len   = info->len;
    csi_rssi      = info->rx_ctrl.rssi;
    csi_timestamp = millis();
    csi_data_ready = true;
}

/* ═══════════════════════════════════════════════════════════
 * PROCESS AND SEND CSI — called from loop()
 * ═══════════════════════════════════════════════════════════ */
void process_and_send_csi() {
    if (!csi_data_ready) return;

    int n_pairs = csi_raw_len / 2;
    if (n_pairs > 256) n_pairs = 256;  // safety cap

    int valid_count = 0;
    int amplitudes[256];

    for (int i = 0; i < n_pairs; i++) {
        int8_t imag = csi_raw_buf[i * 2];
        int8_t real = csi_raw_buf[i * 2 + 1];
        int amp = (int)sqrtf((float)(real * real + imag * imag));
        amplitudes[valid_count] = amp;
        valid_count++;
    }

    if (valid_count == 0) {
        csi_data_ready = false;
        return;
    }

    // Build CSV line: timestamp,rssi,n_subcarriers,amp0,amp1,...
    Serial.print(csi_timestamp);
    Serial.print(',');
    Serial.print(csi_rssi);
    Serial.print(',');
    Serial.print(valid_count);

    for (int i = 0; i < valid_count; i++) {
        Serial.print(',');
        Serial.print(amplitudes[i]);
    }
    Serial.println();

    csi_sent_count++;
    csi_data_ready = false;
}

/* ═══════════════════════════════════════════════════════════
 * SERIAL GPIO COMMAND HANDLER — called from loop()
 * ═══════════════════════════════════════════════════════════ */
void handle_serial_commands() {
    while (Serial.available()) {
        char cmd = Serial.read();

        switch (cmd) {
            case 'G':  // Green LEDs ON, Red OFF
                digitalWrite(PIN_GREEN_1, HIGH);
                digitalWrite(PIN_GREEN_2, HIGH);
                digitalWrite(PIN_RED_1, LOW);
                digitalWrite(PIN_RED_2, LOW);
                break;

            case 'R':  // Red LEDs ON, Green OFF
                digitalWrite(PIN_RED_1, HIGH);
                digitalWrite(PIN_RED_2, HIGH);
                digitalWrite(PIN_GREEN_1, LOW);
                digitalWrite(PIN_GREEN_2, LOW);
                break;

            case 'B':  // Buzzer pulse 500ms (non-blocking)
                digitalWrite(PIN_BUZZER, HIGH);
                buzzer_start_time = millis();
                buzzer_active = true;
                break;

            case 'X':  // All GPIO LOW (reset)
                digitalWrite(PIN_GREEN_1, LOW);
                digitalWrite(PIN_GREEN_2, LOW);
                digitalWrite(PIN_RED_1, LOW);
                digitalWrite(PIN_RED_2, LOW);
                digitalWrite(PIN_BUZZER, LOW);
                buzzer_active = false;
                break;

            case '\n':  // Ignore newline
            case '\r':  // Ignore carriage return
                break;

            default:
                break;
        }
    }
}

/* ═══════════════════════════════════════════════════════════
 * WiFi EVENT HANDLER
 * ═══════════════════════════════════════════════════════════ */
void wifi_event_handler(WiFiEvent_t event) {
    switch (event) {
        case ARDUINO_EVENT_WIFI_STA_GOT_IP:
            Serial.print("[WIFI] Connected — IP: ");
            Serial.println(WiFi.localIP());
            break;
        case ARDUINO_EVENT_WIFI_STA_DISCONNECTED:
            Serial.println("[WIFI] Disconnected — reconnecting...");
            WiFi.reconnect();
            break;
        default:
            break;
    }
}

/* ═══════════════════════════════════════════════════════════
 * SETUP
 * ═══════════════════════════════════════════════════════════ */
void setup() {
    esp_err_t err;

    // 1. Set CPU to 160 MHz for WiFi performance
    setCpuFrequencyMhz(160);

    // 2. Initialize serial at 2 Mbaud
    Serial.begin(SERIAL_BAUD);
    while (!Serial) { delay(10); }
    delay(500);

    // 3. Startup banner
    Serial.println();
    Serial.println("════════════════════════════════════════════");
    Serial.println("=== WiFi Radar — XIAO ESP32-C6 ===");
    Serial.println("Firmware v3.0 — Bistatic Router CSI");
    Serial.println("════════════════════════════════════════════");

    // 4. Configure GPIO pins as outputs, all LOW
    pinMode(PIN_GREEN_1, OUTPUT);
    pinMode(PIN_RED_1,   OUTPUT);
    pinMode(PIN_BUZZER,  OUTPUT);
    pinMode(PIN_RED_2,   OUTPUT);
    pinMode(PIN_GREEN_2, OUTPUT);

    digitalWrite(PIN_GREEN_1, LOW);
    digitalWrite(PIN_GREEN_2, LOW);
    digitalWrite(PIN_RED_1,   LOW);
    digitalWrite(PIN_RED_2,   LOW);
    digitalWrite(PIN_BUZZER,  LOW);

    // 5. Initialize NVS (required for WiFi subsystem)
    err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        Serial.println("[NVS] Erasing and re-initializing...");
        nvs_flash_erase();
        err = nvs_flash_init();
    }
    if (err != ESP_OK) {
        Serial.printf("[NVS] Init error: %d\n", err);
    } else {
        Serial.println("[NVS] OK");
    }

    // 6. Register WiFi event handler
    WiFi.onEvent(wifi_event_handler);

    // 7. Connect to WiFi in STA mode
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASS);

    Serial.print("[WIFI] Connecting to ");
    Serial.print(WIFI_SSID);

    int connect_attempts = 0;
    while (WiFi.status() != WL_CONNECTED && connect_attempts < 40) {
        delay(500);
        Serial.print(".");
        connect_attempts++;
    }
    Serial.println();

    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("[WIFI] Connection FAILED — no CSI source available");
        Serial.println("[WIFI] Check SSID/password and try again");
        // Blink red LEDs to indicate failure
        for (int i = 0; i < 5; i++) {
            digitalWrite(PIN_RED_1, HIGH);
            digitalWrite(PIN_RED_2, HIGH);
            delay(200);
            digitalWrite(PIN_RED_1, LOW);
            digitalWrite(PIN_RED_2, LOW);
            delay(200);
        }
        // Continue anyway — may still capture ambient CSI in promiscuous mode
    } else {
        Serial.print("[WIFI] Connected — IP: ");
        Serial.println(WiFi.localIP());
        Serial.print("[WIFI] Gateway: ");
        Serial.println(WiFi.gatewayIP());
        Serial.print("[WIFI] Channel: ");
        uint8_t primary_chan;
        wifi_second_chan_t second_chan;
        esp_wifi_get_channel(&primary_chan, &second_chan);
        Serial.println(primary_chan);
    }

    // 8. Disable power saving for consistent CSI
    esp_wifi_set_ps(WIFI_PS_NONE);

    // 9. Enable promiscuous mode for broader CSI capture
    err = esp_wifi_set_promiscuous(true);
    if (err != ESP_OK) {
        Serial.printf("[WIFI] Promiscuous mode error: %d\n", err);
    } else {
        Serial.println("[WIFI] Promiscuous mode enabled");
    }

    // 10. Configure CSI with correct ESP32-C6 struct fields
    wifi_csi_config_t csi_config;
    memset(&csi_config, 0, sizeof(csi_config));
    csi_config.enable                 = true;
    csi_config.acquire_csi_legacy     = true;   // L-LTF legacy frames
    csi_config.acquire_csi_ht20       = true;   // HT 20 MHz
    csi_config.acquire_csi_ht40       = true;   // HT 40 MHz
    csi_config.acquire_csi_su         = true;   // HE SU — enable for WiFi 6
    csi_config.acquire_csi_mu         = true;   // HE MU — enable
    csi_config.acquire_csi_dcm        = true;   // HE DCM — enable
    csi_config.acquire_csi_beamformed = true;   // HE beamformed — enable
    csi_config.acquire_csi_he_stbc    = 2;      // int, not bool
    csi_config.val_scale_cfg          = false;
    csi_config.dump_ack_en            = true;   // Also capture CSI from ACK frames
    csi_config.reserved               = false;

    err = esp_wifi_set_csi_config(&csi_config);
    if (err != ESP_OK) {
        Serial.printf("[CSI] Config error: %d\n", err);
    } else {
        Serial.println("[CSI] Configured OK — all frame types enabled");
    }

    // 11. Register CSI callback and enable CSI
    err = esp_wifi_set_csi_rx_cb(wifi_csi_callback, NULL);
    if (err != ESP_OK) {
        Serial.printf("[CSI] Set callback error: %d\n", err);
    }

    err = esp_wifi_set_csi(true);
    if (err != ESP_OK) {
        Serial.printf("[CSI] Enable error: %d\n", err);
    } else {
        Serial.println("[CSI] Enabled");
    }

    // 12. Start ICMP ping to router for continuous traffic at ~100 Hz
    if (WiFi.status() == WL_CONNECTED) {
        esp_ping_config_t ping_config = ESP_PING_DEFAULT_CONFIG();
        ping_config.count           = 0;              // Infinite
        ping_config.interval_ms     = PING_INTERVAL_MS;  // 10ms = 100 Hz
        ping_config.task_stack_size = 3072;
        ping_config.data_size       = 1;              // Smallest payload

        ping_config.target_addr.u_addr.ip4.addr = (uint32_t)WiFi.gatewayIP();
        ping_config.target_addr.type = ESP_IPADDR_TYPE_V4;

        esp_ping_handle_t ping_handle = NULL;
        esp_ping_callbacks_t cbs = { 0 };

        err = esp_ping_new_session(&ping_config, &cbs, &ping_handle);
        if (err != ESP_OK) {
            Serial.printf("[PING] Session error: %d\n", err);
        } else {
            err = esp_ping_start(ping_handle);
            if (err != ESP_OK) {
                Serial.printf("[PING] Start error: %d\n", err);
            } else {
                Serial.printf("[PING] Pinging gateway at %d Hz\n", 1000 / PING_INTERVAL_MS);
            }
        }
    }

    Serial.println();
    Serial.println("READY — streaming CSI");
    Serial.println("FORMAT: timestamp_ms,rssi,n_subcarriers,amp0,amp1,...");

    // 13. Blink green LEDs 3 times to indicate ready
    for (int i = 0; i < 3; i++) {
        digitalWrite(PIN_GREEN_1, HIGH);
        digitalWrite(PIN_GREEN_2, HIGH);
        delay(150);
        digitalWrite(PIN_GREEN_1, LOW);
        digitalWrite(PIN_GREEN_2, LOW);
        delay(150);
    }
}

/* ═══════════════════════════════════════════════════════════
 * MAIN LOOP
 * ═══════════════════════════════════════════════════════════ */
void loop() {
    // Process and send any pending CSI data
    process_and_send_csi();

    // Handle incoming GPIO commands from laptop
    handle_serial_commands();

    // Manage non-blocking buzzer (500ms pulse)
    if (buzzer_active && (millis() - buzzer_start_time >= 500)) {
        digitalWrite(PIN_BUZZER, LOW);
        buzzer_active = false;
    }

    // Debug heartbeat every 5 seconds
    static unsigned long last_heartbeat = 0;
    if (millis() - last_heartbeat >= 5000) {
        Serial.printf("[DEBUG] CSI callbacks: %u, sent: %u, WiFi: %s\n",
                       csi_callback_count, csi_sent_count,
                       WiFi.status() == WL_CONNECTED ? "OK" : "DISCONNECTED");
        last_heartbeat = millis();
    }

    // Small yield to prevent watchdog timeout
    yield();
}
