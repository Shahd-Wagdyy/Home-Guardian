# Home Guardian

**Where Safety Thinks Ahead**

Home Guardian is a hybrid, AI-powered home monitoring platform built to move home safety from "alert after the fact" to "prevent before it happens." It watches over the people (and pets) who matter most — elderly parents, kids, bedridden patients, pets, and empty homes — and reacts to real danger before it becomes a real problem.

This is our B.Sc. graduation project from the **Arab Academy for Science, Technology & Maritime Transport**, College of Computing and Information Technology (Cairo) — Computer Science & Software Engineering.

---

## Why We Built This

Most home monitoring systems today are reactive and fragmented. You get one app for security, another for baby monitoring, another for elderly care, and none of them talk to each other. They also tend to treat everyone the same — but a toddler, a grandparent recovering from surgery, and a curious dog need very different kinds of "watching."

Home Guardian solves this by combining computer vision, machine learning, and behavioral analysis into a **single adaptive engine** that changes what it looks for depending on who's home and what mode is active.

---

## Core Idea: One System, Five Modes

Instead of juggling multiple apps and devices, you just tell Home Guardian who or what needs watching, and it reconfigures its AI models and sensitivity in real time.

| Mode | Focus |
|---|---|
| Silver Mode | Elderly care — fall detection, immobility, daily routine changes |
| Nanny Mode | Child safety — proximity to sharp objects, sockets, restricted zones |
| Nurse Mode | Bedridden patients — prolonged stillness alerts (e.g. pressure ulcer prevention) |
| Pet Mode | Pet safety — monitoring pets near balconies, open windows, hazardous areas |
| Home Alone Mode | Security — detecting unknown individuals when the house is empty |

Switching modes takes effect in under 3 seconds — the AI immediately loads the right rule-set and sensitivity for the situation.

---

## What It Actually Does

- **Real-time hazard prevention** — spots danger before it happens (a baby near a stove, a candle left burning, a pet heading for an open window)
- **Fall and immobility detection** — flags falls and prolonged stillness for elderly or bedridden users
- **Facial recognition access control** — tells the difference between family, registered guests, and strangers (98% accuracy under good lighting)
- **Instant emergency alerts** — pushes notifications to trusted contacts, and escalates to emergency services if no one responds
- **Live and recorded feeds** — view what's happening now, or review past events with timestamped clips
- **Privacy-first design** — local edge processing, encrypted storage, and zone-based privacy masks so the system protects you instead of watching you

---

## Architecture Overview

Home Guardian runs across mobile, web dashboard, and edge hardware, tied together by a FastAPI backend and a PostgreSQL data layer.

```
Camera (Edge) → AI Detection Engine → Backend (FastAPI) → PostgreSQL
                                              |
                                    WebSocket / REST API
                                              |
                          Mobile App (Flutter) & Web Dashboard
                                              |
                              Push Notifications (Firebase) → Trusted Contacts / Emergency Services
```

### Tech Stack

**Frontend (Client)**
- Dart + Flutter (Android, iOS, Web, Desktop — single codebase)
- Provider (state management)
- `fl_chart` for analytics visualizations
- Flutter localization + `Intl` for multi-language support

**Backend**
- Python + FastAPI + Pydantic
- PostgreSQL (via `psycopg2`)
- REST APIs + WebSockets for real-time alerts
- JWT + OAuth2 bearer token authentication, bcrypt password hashing

**AI / Machine Learning**
- TensorFlow / PyTorch
- OpenCV
- YOLO / EfficientDet for real-time object and event detection

**Networking**
- Dual-network architecture: LAN for local access, Tailscale tunneling for secure remote access

**Hardware (Pet Station module)**
- ESP32-CAM
- HC-SR04 ultrasonic sensor, DS3231 RTC, 16x2 LCD, SG90 servo, RC522 RFID reader

**Notifications and Storage**
- Firebase Cloud Messaging
- Secure device storage for tokens/credentials, separate from general app preferences

---

## Security and Privacy

We treat this seriously — the system watches over people's homes, so it has to be trustworthy by design:

- Password hashing with bcrypt
- JWT-based authentication with OAuth2 bearer tokens
- Environment-based secrets management (no hardcoded credentials)
- Clear separation between secure storage (tokens/credentials) and general app settings
- Designed with GDPR and HIPAA principles in mind (data minimization, right to be forgotten, consent flows)

---

## Testing

We didn't just build it — we tried to break it:

- **21 automated unit tests** across backend (pytest) and frontend (Flutter/`mocktail`), covering auth, mode-gating logic, and data serialization
- **16 integration test cases** run via Postman covering the full API surface — auth, monitoring, alerts, and events
- **User Acceptance Testing** through the live dashboard and mobile app, focused on response times, config persistence, and error clarity


## Roadmap

Things we'd love to take further:

- Explainable AI for alert transparency
- Smart medication box integration for elderly/patient care
- Voice-based emergency assistance
- Deeper edge AI processing to reduce cloud dependency
- Healthcare and insurance provider integrations
- Broader smart home ecosystem support (Alexa, Google Home, smart locks)

---

## Team

Built with a lot of coffee and even more debugging by:

- Shahd Wagdy Abdelfattah
- Arwa Ahmed El Mokadem
- Karen Hany Nabil
- Karim Sherif Fathy
- Ahmed Mohamed Mohamed
- Ahmed Osama Ahmed

**Supervised by:** Assoc. Prof. Mohamed Fathy and Dr. Mohamed Ragae

Arab Academy for Science, Technology and Maritime Transport — College of Computing and Information Technology, Cairo.


*Home Guardian — because the best kind of safety is the kind you never have to think about.*
