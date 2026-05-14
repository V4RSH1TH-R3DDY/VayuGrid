# Feature: Hackathon Demo Video Script

**Created:** 2026-05-13  
**Author:** Antigravity (AI Assistant) — based on full codebase audit  
**File:** `docs/VIDEO_SCRIPT.md`

---

## Purpose

A complete, scene-by-scene narration and on-screen action script for recording the VayuGrid
hackathon demo video targeting AI4India 2026 judges. Covers every major system component
with technical accuracy and precise jargon.

---

## Script Structure (12 Scenes)

| Scene | Title | Timestamp | Key Components Covered |
|-------|-------|-----------|----------------------|
| 0 | Cold Open | 0:00–0:30 | Problem hook — 28% solar curtailment stat |
| 1 | Problem & Vision | 0:30–1:30 | Duck Curve, IEEE C57.91 thermal aging, solution overview |
| 2 | Architecture Overview | 1:30–2:15 | Full stack diagram — TimescaleDB, Redis, RabbitMQ, FastAPI, React, MLflow |
| 3 | The Simulator | 2:15–3:00 | `GridSimulator`, `IEEETransformerThermalModel`, `FaultEngine`, Pecan Street data |
| 4 | AI Models | 3:00–4:00 | `CortexCore` PPO agent, `VayuGNN` HGT, `NeighborhoodSignal` schema |
| 5 | P2P Trading Engine | 4:00–4:45 | `MatchingEngine`, `AyaLedger`, Continuous Double Auction |
| 6 | Docker Stack Live | 4:45–5:15 | `docker compose ps` — all 8 containers healthy |
| 7 | Operator Dashboard | 5:15–6:20 | Tony's view — KPIs, Grid Health Map, Duck Curve Tracker, Risk Timeline, Manual Override |
| 8 | Homeowner Dashboard | 6:20–7:10 | Reggie's view — Energy flow, EV status, Earnings delta, DPDP consent/export/deletion |
| 9 | Community Dashboard | 7:10–7:35 | Luigi's view — Backup hours, Community savings, Fairness Allocation |
| 10 | Impact Metrics | 7:35–8:20 | Baseline comparison table, Federated Learning, MARL/MAPPO |
| 11 | Stack Montage | 8:20–8:50 | MLflow, Grafana, AyaLedger.verify(), GridSimulator.run() |
| 12 | Close | 8:50–9:30 | Mission statement, team card |

---

## Appendices Included

- **Appendix A** — Lower-third jargon glossary (14 terms with plain-English definitions)
- **Appendix B** — Per-scene speaker notes with do/don't guidance
- **Appendix C** — Pre-recording checklist (12 items)

---

## Technical Terms Used

- Duck Curve, IEEE C57.91, per-unit (p.u.), transformer aging acceleration
- PPO (Proximal Policy Optimization), curriculum learning, INT8 quantization, ONNX
- HGT (Heterogeneous Graph Transformer), Focal Loss (α=0.75, γ=2.0), temporal self-attention
- Continuous Double Auction, sliding window rate limiting, SHA-256 hash-linked ledger
- DPDP Act 2023, granular consent, data export/deletion endpoints
- TimescaleDB hypertable, Redis live state cache, RabbitMQ mesh bus
- Federated Learning (Flower framework), MARL, MAPPO centralized critic
- WebSocket live-streaming, JWT auth, role-based access control

---

## Credentials Used in Demo

| Dashboard | Username | Password | Role |
|-----------|----------|----------|------|
| Operator | `tony` | `operator` | Grid ops manager |
| Homeowner | `reggie` | `homeowner` | Solar + EV owner |
| Community | `luigi` | `community` | Community organizer |

---

## Notes

- Script was written after auditing all source files: `simulator/simulator.py`,
  `trading/engine.py`, `trading/ledger.py`, `ai/schemas.py`,
  `frontend/src/views/OperatorDashboard.tsx`, `frontend/src/views/HomeownerDashboard.tsx`,
  `frontend/src/views/CommunityDashboard.tsx`, `VayuGrid_Team_Plan.md`,
  `Baseline_Protocol_and_Success_Metrics.md`
- Target runtime: **8–10 minutes**
- Narration is written for voiceover, not teleprompter reading — shorter sentences, natural pauses
