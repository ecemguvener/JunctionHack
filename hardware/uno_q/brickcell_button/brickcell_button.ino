// UNO Q MCU: debounced capture-request button. No arm or camera commands.
#include <Arduino_RouterBridge.h>

constexpr int BUTTON_PIN = 2; // Confirm station wiring before flashing.
unsigned long changedAt = 0;
bool lastRaw = HIGH;
bool stableButton = HIGH;
int requestSequence = 0;
int statusCode = 0;
unsigned long lastStatusAt = 0;

int takeRequest() { return requestSequence; }
void setStatus(int code) {
  statusCode = (code >= 0 && code <= 2) ? code : 0;
  lastStatusAt = millis();
}

void setup() {
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, HIGH); // UNO Q built-in LED is active-low.
  Bridge.begin();
  Bridge.provide("brickcell_take_request", takeRequest);
  Bridge.provide("brickcell_status", setStatus);
}

void loop() {
  bool raw = digitalRead(BUTTON_PIN);
  if (raw != lastRaw) { lastRaw = raw; changedAt = millis(); }
  if (millis() - changedAt >= 50 && raw != stableButton) {
    stableButton = raw;
    if (stableButton == LOW && requestSequence < 2147483647) ++requestSequence;
  }
  if (millis() - lastStatusAt > 3000) statusCode = 0; // Host-loss watchdog.
  // Ready: steady light. Review/error: slow blink. Idle: off.
  bool lit = statusCode == 1 || (statusCode == 2 && (millis() / 500) % 2 == 0);
  digitalWrite(LED_BUILTIN, lit ? LOW : HIGH);
}
