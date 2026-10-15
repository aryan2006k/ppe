/*
  PPE Access Control & Automated Dispenser
  -----------------------------------------
  Architecture:
  1) Mode 1: Evaluation Phase (8 seconds) — Managed by Python software
  2) Mode 2:
     - Access Granted (5 seconds): Green LED ON, then auto-OFF
     - Dispensing Phase (10 seconds): Red LED ON + Missing PPE servo doors OPEN,
       then auto-close after 10s and Red LED auto-OFF

  Wiring:
  - Green LED:        Digital Pin 5
  - Red LED:          Digital Pin 6
  - Gloves Servo:     Digital Pin 9  (Close = 150 deg, Open = 60 deg)
  - Mask Servo:       Digital Pin 10 (Close = 110 deg, Open = 10 deg)
  - Baud Rate:        9600
*/

#include <Servo.h>

Servo glovesServo;
Servo maskServo;

// Pin assignments
const int GREEN_LED = 5;
const int RED_LED   = 6;
const int GLOVES_SERVO_PIN = 9;
const int MASK_SERVO_PIN   = 10;

// Servo calibrated angles
const int GLOVES_CLOSE = 150;
const int GLOVES_OPEN  = 60;
const int MASK_CLOSE   = 110;
const int MASK_OPEN    = 10;

// Phase 2 Hold Durations (matching Python unified_config.py)
const unsigned long GRANT_HOLD_MS    = 5000UL;   // 5 seconds green LED
const unsigned long DISPENSE_HOLD_MS = 10000UL;  // 10 seconds red LED + open doors

// Non-blocking timer states
bool greenLedActive   = false;
unsigned long greenLedStart = 0;

bool redLedActive     = false;
unsigned long redLedStart   = 0;

bool glovesDoorOpen   = false;
unsigned long glovesDoorStart = 0;

bool maskDoorOpen     = false;
unsigned long maskDoorStart   = 0;

void setup() {
  Serial.begin(9600);
  Serial.setTimeout(50);

  glovesServo.attach(GLOVES_SERVO_PIN);
  maskServo.attach(MASK_SERVO_PIN);

  pinMode(GREEN_LED, OUTPUT);
  pinMode(RED_LED,   OUTPUT);

  // Initial resting state: Doors closed, LEDs off
  glovesServo.write(GLOVES_CLOSE);
  maskServo.write(MASK_CLOSE);
  digitalWrite(GREEN_LED, LOW);
  digitalWrite(RED_LED,   LOW);

  Serial.println("READY");
}

void loop() {
  unsigned long now = millis();

  // -------------------------------------------------------------
  // Mode 2a: Auto-turn off Green LED after 5 seconds
  // -------------------------------------------------------------
  if (greenLedActive && (now - greenLedStart >= GRANT_HOLD_MS)) {
    digitalWrite(GREEN_LED, LOW);
    greenLedActive = false;
    Serial.println("[ARDUINO] Green LED OFF (5s complete)");
  }

  // -------------------------------------------------------------
  // Mode 2b: Auto-turn off Red LED after 10 seconds
  // -------------------------------------------------------------
  if (redLedActive && (now - redLedStart >= DISPENSE_HOLD_MS)) {
    digitalWrite(RED_LED, LOW);
    redLedActive = false;
    Serial.println("[ARDUINO] Red LED OFF (10s complete)");
  }

  // -------------------------------------------------------------
  // Mode 2b: Auto-close Gloves servo door after 10 seconds
  // -------------------------------------------------------------
  if (glovesDoorOpen && (now - glovesDoorStart >= DISPENSE_HOLD_MS)) {
    glovesServo.write(GLOVES_CLOSE);
    glovesDoorOpen = false;
    Serial.println("[ARDUINO] Gloves door closed (10s complete)");
  }

  // -------------------------------------------------------------
  // Mode 2b: Auto-close Mask servo door after 10 seconds
  // -------------------------------------------------------------
  if (maskDoorOpen && (now - maskDoorStart >= DISPENSE_HOLD_MS)) {
    maskServo.write(MASK_CLOSE);
    maskDoorOpen = false;
    Serial.println("[ARDUINO] Mask door closed (10s complete)");
  }

  // -------------------------------------------------------------
  // Non-blocking Serial Command Receiver
  // -------------------------------------------------------------
  if (!Serial.available()) return;

  String command = Serial.readStringUntil('\n');
  command.trim();
  if (command.length() == 0) return;

  if (command == "GRANT") {
    // Mode 2a: Access Granted -> Green LED ON for 5 seconds
    digitalWrite(RED_LED,   LOW);
    digitalWrite(GREEN_LED, HIGH);
    greenLedActive = true;
    greenLedStart  = millis();
    redLedActive   = false;
    Serial.println("[ARDUINO] Mode 2a: Access Granted -> Green LED ON (5s)");

  } else if (command == "DENY") {
    // Mode 2b: Not Validated -> Red LED ON for 10 seconds
    digitalWrite(GREEN_LED, LOW);
    digitalWrite(RED_LED,   HIGH);
    redLedActive = true;
    redLedStart  = millis();
    greenLedActive = false;
    Serial.println("[ARDUINO] Mode 2b: Not Validated -> Red LED ON (10s)");

  } else if (command == "GLOVES") {
    // Mode 2b: Open gloves door for 10 seconds
    glovesServo.write(GLOVES_OPEN);
    glovesDoorOpen  = true;
    glovesDoorStart = millis();
    Serial.println("[ARDUINO] Mode 2b: Gloves door OPEN (10s)");

  } else if (command == "MASK") {
    // Mode 2b: Open mask door for 10 seconds
    maskServo.write(MASK_OPEN);
    maskDoorOpen  = true;
    maskDoorStart = millis();
    Serial.println("[ARDUINO] Mode 2b: Mask door OPEN (10s)");

  } else if (command == "RESET") {
    // Force immediate reset to IDLE resting state
    digitalWrite(GREEN_LED, LOW);
    digitalWrite(RED_LED,   LOW);
    glovesServo.write(GLOVES_CLOSE);
    maskServo.write(MASK_CLOSE);
    greenLedActive = false;
    redLedActive   = false;
    glovesDoorOpen = false;
    maskDoorOpen   = false;
    Serial.println("[ARDUINO] Reset: All doors closed, LEDs OFF");
  }
}
