# VayuGrid — Demo Video Script

> **Setup:** Have the terminal ready with `PYTHONPATH=$PWD python3 scripts/demo_runner.py` staged but not yet run. Have `http://localhost:5173` open in the browser, logged in as `tony` / `operator`.

---

## 0. Opening (15 sec)

**Visual:** Screen shows VayuGrid logo / title card.

**VO:**
> "Welcome to VayuGrid — an intelligent energy grid platform that combines grid simulation, AI-driven control, peer-to-peer energy trading, and real-time monitoring into one integrated system. In this demo, we'll walk through every layer end-to-end."

---

## 1. Grid Simulation Engine (45 sec)

**Visual:** Terminal — run `demo_runner.py`, show simulation phase output.

**VO:**
> "We start with the Grid Simulation Engine. VayuGrid models a residential neighborhood with solar panels, batteries, and electric vehicles. Here we're running the demo scenario — ten homes over a two-hour window, with 67% solar and battery adoption and 33% EV adoption."

**VO:**
> "The scenario also injects a fault — a cloud transient at 11 AM that drops 75% of solar output for 20 minutes. The simulator computes load flows, battery dispatch, transformer thermal dynamics, and voltage across every node in the network. As you can see, the simulation completes in under a second, producing over 2,000 data points."

---

## 2. Database Ingestion (30 sec)

**Visual:** Terminal output showing DB insertion counts.

**VO:**
> "These results don't just stay in memory — VayuGrid ingests them into TimescaleDB, a time-series database built on PostgreSQL. Telemetry data, transformer readings, and simulated peer-to-peer trades are all written to the database in real time."

**VO:**
> "This is the same database that powers the live dashboards and the WebSocket stream. Once the data is in, the frontend immediately has something to display."

---

## 3. AI Agent Inference (60 sec)

**Visual:** Terminal showing agent running, action columns, reward output.

**VO:**
> "Now let's look at the AI layer. VayuGrid's CortexCore agent is a Proximal Policy Optimization — or PPO — model trained to manage battery dispatch, EV charging, and grid trading decisions."

**VO:**
> "The agent loads a trained checkpoint and runs against the simulator environment. For each time step, it outputs five actions: battery charge or discharge rate, EV charging rate, bid price, ask price, and grid import-export. The policy column shows whether the trained PPO model is making the decision, or whether the system has fallen back to a rule-based controller."

**VO:**
> "This fallback mechanism is critical for production systems — if the model produces unreasonable actions, the RuntimeWithFallback wrapper seamlessly switches to a safe baseline without interrupting operations."

---

## 4. REST API Layer (45 sec)

**Visual:** Terminal output from API endpoint probes. Then switch to browser with API request in dev tools or curl.

**VO:**
> "All of this data is accessible through VayuGrid's REST API. Let's probe a few endpoints. The health check confirms the service is running. Authentication uses JWT tokens — we log in as the operator role."

**VO:**
> "The operator dashboard overview returns KPIs like curtailment percentage, peak reduction, and overload events, along with a real-time grid health snapshot for every node in the neighborhood."

**VO:**
> "There are separate endpoints for the homeowner view — showing individual costs, earnings, and self-sufficiency — and the community view, which tracks shared savings and fairness scores across the neighborhood."

---

## 5. WebSocket Live Stream (30 sec)

**Visual:** Terminal showing WebSocket connection and live data. Then switch to browser with frontend open.

**VO:**
> "For real-time updates, VayuGrid provides a WebSocket stream that pushes a complete system snapshot every five seconds. The stream includes the latest telemetry from every node, recent trade settlements, and GNN-based overload probability predictions for each transformer."

**VO:**
> "This is what powers the live-updating dashboards in the frontend. As new data arrives, the charts and KPIs refresh automatically without any page reload."

---

## 6. Frontend Dashboards (45 sec)

**Visual:** Browser — navigate through Operator, Homeowner, and Community dashboards.

**VO:**
> "Let's log into the frontend. We have three user roles: Operator, Homeowner, and Community. Each sees a dashboard tailored to their needs."

**VO (Operator view):**
> "The Operator dashboard shows the full grid overview — loading curves, voltage across nodes, transformer risk timelines, and the duck curve from solar generation. From here, operators can issue neighborhood signals like throttle requests or islanding commands."

**VO (Homeowner view):**
> "The Homeowner dashboard shows personal energy consumption, solar generation, battery status, and trading activity. Homeowners can manage their data consent preferences and see how much they've saved through peer-to-peer trades."

**VO (Community view):**
> "The Community dashboard focuses on collective metrics — total shared savings, fairness of energy distribution, and critical load flags for vulnerable households."

---

## 7. Closing (15 sec)

**Visual:** Fade back to terminal with demo completed message.

**VO:**
> "That's VayuGrid — from grid simulation through AI control to live dashboards, all working together as one platform. Every layer is open source and designed for real-world smart grid deployments. Thank you for watching."

---

## Quick Reference

| Timestamp | Section | On Screen |
|-----------|---------|-----------|
| 0:00 | Opening | Title card |
| 0:15 | Simulation | Terminal — simulation output |
| 1:00 | Database | Terminal — DB insertion |
| 1:30 | AI Agent | Terminal — agent inference output |
| 2:30 | REST API | Terminal + browser — API probes |
| 3:15 | WebSocket | Terminal + browser — live stream |
| 3:45 | Dashboards | Browser — three dashboards |
| 4:30 | Closing | Terminal — demo complete |

**Total runtime: ~4 minutes 45 seconds**
