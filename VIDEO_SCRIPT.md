`# VayuGrid — Hackathon Demo Video Script
### AI4India Hackathon | Team Coder4not4
**Total Target Runtime:** 8–10 minutes  
**Tone:** Confident, technical, awestruck-worthy — not salesy. Let the engineering do the talking.

---

## PRE-PRODUCTION NOTES

- **Screen resolution:** 1920×1080, browser at 100% zoom, dark OS theme
- **Voiceover pace:** Slightly slower than conversational — judges will be reading + listening
- **On-screen cue format:** `[ACTION]` = what you do on screen | `[SHOW]` = what should be visible
- **Transitions:** Clean cuts only. No wipes, no sparkles. This is infrastructure.
- **Music:** Ambient low-pulse electronic (no vocals). Fade under narration, kill during live demos.

---

## SCENE 0 — COLD OPEN (0:00–0:30)

**[SHOW: Black screen. Single line of white text appears, typewriter style:]**

> *"India wastes 28% of every unit of solar energy generated in residential neighborhoods."*

**[NARRATION — over black screen]**

> "Twenty-eight percent. Not because the panels fail. Not because there aren't enough batteries.
> Because the grid — as it exists today — has no intelligence. No coordination. No way for your
> rooftop solar to talk to your neighbor's EV charger, or to know that the distribution transformer
> two streets over is about to trip. Every kilowatt-hour wasted is money burned and carbon emitted."

**[SHOW: Text fades. New text appears:]**

> *"VayuGrid. Intelligent peer-to-peer energy for Indian neighborhoods."*

**[CUT TO: Team intro card — names, roles, hackathon name]**

---

## SCENE 1 — PROBLEM & VISION (0:30–1:30)

**[SHOW: A simple animated diagram — a neighborhood with homes, a transformer in the middle, solar panels on rooftops, EV chargers in driveways. Draw the "Duck Curve" as a graph overlay — net load dips midday (solar peak) then spikes at 6 PM evening peak.]**

**[NARRATION]**

> "The problem has a name: the Duck Curve. Midday, solar floods the grid. Load drops.
> The distribution transformer — already decades old — gets stressed by the reverse power flow.
> Then at 6 PM, everyone turns on their AC, their TV, their EV charger. Load spikes.
> The transformer gets hit again. And again. And again. Every day."

**[SHOW: Annotate the diagram — highlight the transformer heating up, 'thermal aging acceleration']**

> "Under IEEE C57.91 standard, every degree above rated temperature *doubles* the transformer's
> aging rate. You're not just wasting solar. You're quietly killing a piece of infrastructure
> that costs lakhs of rupees to replace."

**[NARRATION]**

> "VayuGrid solves this with two things: a Graph Neural Network that *sees* the neighborhood
> as an electrical graph and predicts failures before they happen — and a Reinforcement Learning
> agent running on each home's Vayu-Node device, autonomously managing batteries, solar, and EV
> charging every single minute. No cloud dependency. No human in the loop."

---

## SCENE 2 — ARCHITECTURE OVERVIEW (1:30–2:15)

**[SHOW: System architecture diagram — draw/animate the following layers top to bottom:]**

```
+----------------------------------------------------------+
|            REACT DASHBOARDS (Vite + TypeScript)          |
|      Operator | Homeowner (PWA) | Community (Luigi)      |
+-------------------------+--------------------------------+
                          | WebSocket + JWT REST
+-------------------------v--------------------------------+
|            FastAPI Backend — Port 8000                   |
|    Auth (JWT/API-key) . Rate limiting . WebSocket /ws    |
+-------+------------------+-------------------------------+
        |                  |
+-------v------+   +-------v-------+   +------------------+
| TimescaleDB  |   |     Redis     |   |    RabbitMQ      |
| (Hypertable  |   | (Live State   |   |  (Mesh Message   |
|  Telemetry)  |   |  Cache)       |   |   Bus)           |
+--------------+   +---------------+   +------------------+
        |                                       |
+-------v---------------------------------------v----------+
|                   CORE ENGINE LAYER                      |
|  GridSimulator . MatchingEngine . AyaLedger . VayuGNN   |
|  CortexCore (PPO Agent) . FaultEngine . ThermalModel    |
+----------------------------------------------------------+
        |
+-------v--------------------------------------------------+
|               MLflow + Prometheus + Grafana              |
|      Experiment tracking . Metrics . Observability       |
+----------------------------------------------------------+
```

**[NARRATION — brisk, confident, name-drop every component]**

> "The stack. Bottom-up. The foundation is TimescaleDB — PostgreSQL with a time-series
> hypertable extension — storing every node's minute-resolution telemetry. Redis caches live
> state for sub-millisecond lookup by the matching engine. RabbitMQ is the message bus that
> carries trade orders and neighborhood signals across the mesh."

> "Above that: our core engine layer. A fully functional grid simulator, the P2P Continuous
> Double Auction matching engine, the Aya Wallet Ledger — a hash-linked, tamper-evident trade
> record — and the two AI models: CortexCore, the PPO reinforcement learning agent that runs
> on each home device, and VayuGNN, the Heterogeneous Graph Transformer that watches the
> whole neighborhood."

> "All surfaced through FastAPI with JWT auth, WebSocket live-streaming, and three purpose-built
> React dashboards. MLflow tracks every training experiment. Prometheus and Grafana give us
> full observability. Everything spins up with a single `docker compose up`."

---

## SCENE 3 — THE SIMULATOR (2:15–3:00)

**[SHOW: Terminal window. Navigate to the project root.]**

```bash
ls simulator/
```

**[SHOW: Output — simulator.py, config.py, faults.py, graph.py, load_profiles.py, thermal.py]**

**[NARRATION]**

> "Before we trained anything, we had to build the virtual world. The VayuGrid Grid Simulator."

**[SHOW: Open simulator/simulator.py — scroll to GridSimulator.__init__ where ResidentialFeederGraph, IEEETransformerThermalModel, and FaultEngine are instantiated]**

> "This is GridSimulator. Every run instantiates a neighborhood as a real electrical network —
> homes and batteries are nodes, distribution lines are edges with real resistance and ampacity
> constraints. The ResidentialFeederGraph solves power flow equations every minute."

**[SHOW: Scroll to IEEETransformerThermalModel instantiation]**

> "The transformer isn't a black box. We implement the IEEE C57.91 thermal model — tracking
> top-oil temperature rise, hotspot rise, and computing an aging acceleration factor every
> timestep. When loading exceeds 1.2 per-unit for five consecutive minutes, we log a
> transformer overload event with a 10-minute cooldown window. This is textbook power
> engineering, implemented in code."

**[SHOW: Scroll to FaultEngine instantiation and the fault types: FAULT_OVERLOAD, FAULT_SOLAR_DROPOUT, FAULT_GRID_OUTAGE, FAULT_PLANNED_MAINTENANCE]**

> "And the fault engine injects real-world scenarios: sudden solar dropouts from cloud transients,
> transformer overload stress, grid outages triggering island mode — and, critically, planned
> maintenance windows that *look* like outages but must NOT trigger automated responses.
> A model that can't distinguish these is dangerous to deploy."

**[SHOW: load_profiles.py briefly — city profile section]**

> "Training data uses Pecan Street 1-minute residential energy data, scaled to Indian consumption
> patterns — 5 to 8 kWh per household per day — with NREL NSRDB solar curves substituted for
> Bengaluru, Chennai, Delhi, Kochi, Hyderabad. Domain randomization varies battery capacity
> by 20%, panel efficiency by up to 15%, and simulates 50 to 500 millisecond message delays on
> market signals. This is what makes a trained agent robust, not just accurate on training data."

---

## SCENE 4 — THE AI MODELS (3:00–4:00)

**[SHOW: ai/schemas.py — pan over the four dataclasses: NodeState, TradeOrder, GridTelemetry, NeighborhoodSignal]**

**[NARRATION]**

> "Four data contracts power the entire AI layer."

> "NodeState — what a single home looks like at any moment: battery SoC in kWh, solar output,
> household load, EV charge level, hours to EV deadline, net grid power, voltage per-unit,
> current buy and sell prices, and — this is important — 15-minute-ahead forecasts for solar,
> load, and price. The agent always has a look-ahead window."

> "NeighborhoodSignal — the GNN's broadcast commands: THROTTLE, PRE_COOL, ISLAND, RESUME.
> With severity, target node IDs, and recommended price floors and caps that dynamically
> reshape the market during grid stress."

**[SHOW: Diagram of CortexCore — asymmetric actor-critic: large critic for training, small actor for deployment]**

**[NARRATION — CortexCore]**

> "CortexCore — the per-home PPO reinforcement learning agent. Trained with curriculum
> learning: start with a single home, no market. Add P2P market at epoch 500. Multi-node
> environment with fault injection at epoch 1,500. Real Pecan Street data fine-tuning
> after 3,000 epochs."

> "The reward function is carefully balanced: maximize P2P trade profit and bill reduction,
> penalize battery SoC violations — operating below 10% or above 95% degrades cells —
> penalize missing EV deadline with a hard penalty, and penalize ignoring THROTTLE signals
> from the GNN. The actor is a small two-layer network — small enough to compute in under
> 5 milliseconds on a Raspberry Pi 5, exported as INT8-quantized ONNX."

**[SHOW: Diagram of VayuGNN — spatial HGT stage feeding into temporal multi-head self-attention stage]**

**[NARRATION — VayuGNN]**

> "VayuGNN — the neighborhood brain. A Heterogeneous Graph Transformer — heterogeneous
> because homes and transformers are different node types with different feature sets and
> different roles. The HGT spatial stage learns message passing: a transformer seeing high
> load tells its connected homes. Homes with surplus solar tell their neighbors."

> "Then multi-head self-attention across 12 consecutive graph snapshots — the temporal stage
> — learns which 12-minute patterns precede failures. Trained with Focal Loss, alpha 0.75,
> gamma 2.0, to handle the severe class imbalance — overloads are rare, but catastrophically
> important. The false positive rate on island signals is our hardest constraint: under 1%.
> A false island disconnects a neighborhood. That is not a metric. That is a safety condition."

---

## SCENE 5 — THE P2P TRADING ENGINE (4:00–4:45)

**[SHOW: trading/engine.py — the MatchingEngine class, submit_order, settle_once methods]**

**[NARRATION]**

> "The P2P trading layer. A Continuous Double Auction — the same market mechanism used by
> most financial exchanges. Every 10 seconds, the matching engine checks if the best bid
> is at or above the best ask. If so — trade at the midpoint price."

**[HIGHLIGHT: submit_order — the rate limiting check and market_state.clamp_price call]**

> "Anti-manipulation is built in. Orders are clamped to dynamic price floors and ceilings
> set by the GNN based on grid stress. Rate limiting: 10 orders per node per minute — enforced
> with a sliding window deque. Orders expire after 60 seconds. No single order can exceed
> 5 kWh. Any node exceeding the rate limit gets rejected instantly, not queued."

**[SHOW: trading/ledger.py — the AyaLedger.append and AyaLedger.verify methods]**

> "Every settled trade is recorded in the Aya Wallet Ledger — a hash-linked,
> append-only log. Each entry SHA-256 hashes the previous entry's hash. If anyone
> tampers with a single record, the entire chain fails verification. No blockchain,
> no mining, no consensus protocol — you don't need distributed consensus for a
> 200-home neighborhood. You need cryptographic integrity. That's exactly what this is."

**[SHOW: The GENESIS_HASH = '0' * 64 line]**

> "And yes — it starts with sixty-four zeros."

---

## SCENE 6 — LIVE DEMO: DOCKER STACK UP (4:45–5:15)

**[SHOW: Terminal — project root]**

```bash
docker compose ps
```

**[SHOW: All containers in healthy/running state — timescaledb, redis, rabbitmq, mlflow, prometheus, grafana, api, frontend]**

**[NARRATION]**

> "Everything up. TimescaleDB, Redis, RabbitMQ, MLflow, Prometheus, Grafana, the FastAPI
> backend, the React frontend — single docker compose up --build. Let's navigate to
> the dashboard."

---

## SCENE 7 — LIVE DEMO: OPERATOR DASHBOARD (5:15–6:20)

**[ACTION: Open browser → http://localhost:5173]**
**[ACTION: Login with tony / operator]**

**[SHOW: Login screen, then redirect to Operator Dashboard]**

**[NARRATION]**

> "We're logged in as Tony — the grid operations manager. JWT authenticated, role-based
> access control routing him straight to the Operator view."

**[SHOW: KPI stat cards at the top — Curtailment, Peak Reduction, Transformer Aging Rate, P2P Volume, P2P Value, Overload Events]**

> "Top of screen: the KPIs that matter. Solar curtailment percentage. Peak demand reduction
> versus the no-control baseline. Transformer aging acceleration rate. P2P volume in
> kilowatt-hours and rupees. And overload events — our target is under 2 per month.
> These aren't mock numbers — they're computed live from the TimescaleDB hypertable."

**[SHOW: Grid Health Map — node cards with stress_level badges: low/medium/high, voltage_pu readings]**

> "The Grid Health Map. Every Vayu-Node in the neighborhood, color-coded by stress level.
> Voltage per-unit. Household load. The GNN's risk signal is embedded in this view —
> nodes where the GNN has elevated risk show as 'high' stress before the transformer
> physically overloads. This is proactive grid management."

**[SHOW: Duck Curve Tracker chart — two lines: actual_kw (blue) and forecast_kw (purple)]**

> "The Duck Curve Tracker. Blue is actual net neighborhood load. Purple is VayuGNN's
> 24-hour forecast. Watch how the forecast predicts the evening ramp. Tony uses this
> to pre-position battery dispatch and issue PRE_COOL signals before the peak, not after it."

**[SHOW: Risk Timeline chart — overload_probability scrolling in real time, red line]**

> "Risk Timeline. The GNN's transformer overload probability, live, next 30 minutes.
> The moment this crosses 0.85 — ISLAND signal fires automatically."

**[ACTION: In Manual Override Panel — set Signal Type to 'THROTTLE', Severity to '0.7', Targets to '1,2,3', Reason to 'Pre-peak load reduction'. Click 'Broadcast Signal'.]**

**[SHOW: Signal queued confirmation message appears]**

> "And the Manual Override Panel. Tony can issue any signal — THROTTLE, PRE_COOL, ISLAND, RESUME —
> to any subset of nodes, with a custom severity and reason logged to the signal history.
> Because the AI will sometimes be wrong, and the human must always be able to take control."

**[SHOW: Signal History table — the just-sent signal appears in the log]**

> "Audit trail. Every signal, every timestamp, every target node set. Immutable log."

---

## SCENE 8 — LIVE DEMO: HOMEOWNER DASHBOARD (6:20–7:10)

**[ACTION: Logout. Login as reggie / homeowner]**

**[SHOW: Homeowner Dashboard — Today's Energy stat cards: solar_generated_kwh, home_consumed_kwh, grid_imported_kwh, p2p_sold_kwh, p2p_bought_kwh, net_bill_inr]**

**[NARRATION]**

> "Reggie's view. A Progressive Web App — mobile-first. Today's energy flow at a glance.
> How much solar was generated. How much was consumed at home. How much was sold to neighbors
> via P2P. How much was imported from the grid. And the number Reggie actually cares about:
> net bill impact in rupees."

**[SHOW: EV Status card — progress bar filling up, current_kwh, target_kwh, deadline]**

> "EV status. The CortexCore agent's current charging plan. Progress bar showing current
> state of charge versus target. Deadline visible. The agent plans charging dynamically —
> it will shift energy-intensive charging to 2-4 AM when P2P prices will be lowest, and the
> homeowner can see that plan happening in real time."

**[SHOW: Battery Health card — SoC, capacity, health_pct]**

> "Battery health. SoC in kWh, capacity, health percentage. The agent respects SoC bounds —
> never discharging below 10% unless in emergency islanding, never charging above 95%.
> Not because we told it the rules — because violating those bounds is penalized in
> the reward function."

**[SHOW: Earnings card — p2p_revenue_inr, delta_vs_net_metering_inr]**

> "Earnings. P2P revenue versus what standard net metering would have paid. The delta
> is the quantified value of VayuGrid's intelligence — the rupees Reggie earns specifically
> because his solar surplus was sold to a neighbor at a better price than the utility buys
> it back at. This number is the ROI justification for every rooftop solar owner."

**[SHOW: Consent & Privacy section — checkboxes, Save Consent, Download Data, Request Deletion buttons]**

**[NARRATION — slow down here, this is important]**

> "Privacy, by design. Not as a checkbox. The Digital Personal Data Protection Act 2023
> imposes specific obligations on any system processing Indian citizens' personal data.
> VayuGrid implements all of them. Granular consent by data category — telemetry, market,
> device, billing — each independently opt-in."

**[ACTION: Click 'Download Data' button]**

**[SHOW: JSON file downloads — vayugrid_node_1_export.json]**

> "One click. Every data point ever stored about this node — exported as a verifiable JSON.
> This is a legal right under DPDP, and it works."

**[ACTION: Click 'Request Deletion']**

**[SHOW: Deletion scheduled confirmation: 'Deletion scheduled for [timestamp]']**

> "Deletion request. Scheduled within 72 hours. Raw 1-minute readings auto-delete after 90
> days at the database level. This isn't future work. It's implemented."

---

## SCENE 9 — LIVE DEMO: COMMUNITY DASHBOARD (7:10–7:35)

**[ACTION: Logout. Login as luigi / community]**

**[SHOW: Community Dashboard — large header: 'Your community currently has X hours of backup power.']**

**[NARRATION]**

> "Luigi's view. Completely different design language. Luigi's residents are not engineers.
> The most important thing they need to know: how long can the neighborhood survive an outage?
> That number is live. It updates every 5 seconds via WebSocket."

**[SHOW: Community Savings panel — today, this month, since installation]**

> "Community savings versus utility rates. Today. This month. Total since install.
> These are real rupee figures — not percentages, not abstractions. Real money."

**[SHOW: Fairness Allocation panel — priority queue display]**

> "And during islanding — when the grid is down and battery reserves are rationed — the
> Fairness Allocation panel makes the priority queue visible. Medical-critical loads first:
> CPAP machines, oxygen concentrators, refrigerated medication. Then food safety. Then
> basic lighting. Then communications. Then comfort, distributed proportionally.
> Every resident can see why these decisions are being made."

---

## SCENE 10 — IMPACT METRICS (7:35–8:20)

**[SHOW: Clean slide — Baseline vs VayuGrid numbers side by side]**

| Metric | No-AI Baseline | VayuGrid Target |
|--------|----------------|-----------------|
| Solar curtailment | 28% | < 5% |
| Peak demand reduction | 0% | > 25% |
| Transformer overloads/month | ~15 events | < 2 events |
| Household electricity cost reduction | 0% | > 20% |
| Islanding switchover time | Full blackout | < 2 minutes |
| P2P trade settlement latency p95 | N/A | < 500 ms |

**[NARRATION — let the numbers breathe]**

> "These are not optimistic projections. These are locked baseline criteria — defined in code,
> measured against three controllers: B0 with no automation, B1 with rule-based dispatch,
> and B2 with Model Predictive Control. VayuGrid's PPO agent must beat all three by at
> least 20% on total cost to ship."

> "Solar curtailment below 5% — that is the surplus that used to be wasted, now traded
> to neighbors at market price. Peak demand down 25% — that is the transformer aging
> rate dropping by half or more. Under 2 overload events per month — that is years of
> additional transformer life, deferred capital expenditure for the utility."

> "Settlement under 500 milliseconds. We target p95 under 200. The Continuous Double
> Auction engine doesn't keep anyone waiting."

**[SHOW: Scale diagram — one neighborhood to federated neighborhoods to regional aggregator]**

> "And it scales. Federated learning with the Flower framework means the GNN learns
> from every neighborhood — without any raw data ever leaving its origin site. DPDP
> compliant by architecture. Multi-Agent RL with MAPPO and a centralized critic turns
> independent home agents into emergent neighborhood cooperators — duck curve flattening
> without explicit coordination."

---

## SCENE 11 — THE STACK IN FULL (8:20–8:50)

**[SHOW: Rapid montage — 3 seconds each:]**

1. `docker compose ps` — all 8 containers healthy
2. MLflow UI — training runs, metrics, model registry
3. Grafana dashboard — RabbitMQ queue depths, API latency heatmap
4. TimescaleDB hypertable continuous aggregate query running fast
5. The `AyaLedger.verify()` returning `True` — SHA-256 chain intact
6. The `GridSimulator.run()` function — 1,440 timesteps (one full day) completing

**[NARRATION]**

> "TimescaleDB for time-series at scale. Redis for microsecond state lookup. RabbitMQ
> for reliable mesh messaging. MLflow for reproducible AI experiments. Prometheus and
> Grafana for every metric, every service, real time. FastAPI for a clean, documented,
> JWT-secured API. React with WebSocket live-streaming for dashboards that never go stale."

> "It runs. It is tested. It is documented. The simulator produces month-long
> neighborhood histories in seconds. The matching engine settles trades in under 500
> milliseconds. The Aya Ledger verifies its entire chain in one call."

---

## SCENE 12 — CLOSE (8:50–9:30)

**[SHOW: Black screen. Text appears:]**

> "28 trillion units of electricity are generated in India annually.
> An estimated 6-8% is lost to inefficiency in the distribution last mile.
> VayuGrid addresses that last mile — one neighborhood at a time."

**[NARRATION — quieter, genuine]**

> "India is adding gigawatts of rooftop solar every year. The distribution grid was not
> designed for this. Transformers are being stressed in ways they were never engineered for.
> Utilities cannot upgrade infrastructure fast enough. The problem needs to be solved at
> the edge — intelligently, autonomously, and fairly."

> "VayuGrid is not a prototype of an idea. It is a working prototype of a system.
> The simulator runs. The trading engine settles. The ledger verifies. The dashboards
> stream live. The AI models are trained and quantized. The DPDP compliance is implemented,
> not promised."

> "We are team Coder4not4. We built VayuGrid. And this is what India's energy future looks
> like when the grid gets a brain."

**[SHOW: Final card]**

> VayuGrid — AI4India Hackathon 2026
> Mukul . Charithra . Varshith . Anjali

---

## APPENDIX A — JARGON GLOSSARY FOR ON-SCREEN LOWER THIRDS

Use these as text overlays when first introducing each term:

| Term | Lower Third Text |
|------|-----------------|
| Duck Curve | "Net load shape caused by high solar penetration — midday dip, evening spike" |
| IEEE C57.91 | "International transformer thermal modeling standard — aging acceleration formula" |
| PPO | "Proximal Policy Optimization — state-of-the-art policy gradient RL algorithm" |
| HGT | "Heterogeneous Graph Transformer — handles multi-type nodes in a graph" |
| Focal Loss | "Loss function that upweights rare events — prevents model ignoring overloads" |
| Continuous Double Auction | "Market mechanism used by most global exchanges — bid/ask matching at midpoint" |
| MARL / MAPPO | "Multi-Agent RL with centralized critic — emergent cooperation without communication" |
| DPDP Act 2023 | "Digital Personal Data Protection Act — India's national data privacy law" |
| Hypertable | "TimescaleDB's time-partitioned table — keeps queries fast at billion-row scale" |
| Federated Learning | "Train on distributed data; share only gradients, not raw data" |
| AyaLedger | "VayuGrid's hash-linked append-only trade record — tamper-evident without blockchain" |
| INT8 Quantization | "Compress neural network to 8-bit integers — 4x smaller, <1% accuracy loss" |
| per-unit (p.u.) | "Normalized voltage/power — 1.0 p.u. = nominal, 1.2 p.u. = 20% overload" |
| Island Mode | "Grid-disconnected neighborhood operating on local solar + batteries alone" |

---

## APPENDIX B — SPEAKER NOTES BY SECTION

### For Tony's Dashboard demo (Scene 7):
- **Do NOT** trigger a live ISLAND signal unless you have a recovery plan — explain that the threshold is 0.85 risk score
- **Do** hover over the KPI cards and let the live WebSocket tick update the numbers in real time
- **Do** pause on the Risk Timeline chart for 3+ seconds — let judges see it's actually live, not a screenshot

### For Reggie's Dashboard demo (Scene 8):
- **Do** actually click Download Data and show the JSON file — this proves DPDP is implemented
- **Do** explain that the EV deadline and charging schedule shown are computed by CortexCore in real time, not hardcoded
- **Do NOT** skip the earnings delta — this is the clearest ROI number in the entire demo

### For Luigi's Dashboard demo (Scene 9):
- **If islanding is not active:** explain the Fairness Allocation panel in theory — "during a grid outage, this panel becomes active and shows exactly this priority hierarchy"
- **Do** say "no jargon, no charts, just: how many hours does your community have" — this is the accessibility design philosophy

### When showing code (Scenes 3-5):
- Terminal font size: 16px minimum
- Use `bat` or VS Code for syntax highlighting — makes code more readable on video
- **Never** scroll faster than judges can read — 1 second per recognizable line minimum

---

## APPENDIX C — RECORDING CHECKLIST

- [ ] Docker stack fully up and healthy before recording starts
- [ ] All three dashboards tested and returning real data from API
- [ ] WebSocket stream confirmed live (status indicator visible on-screen)
- [ ] MLflow has at least 3 training runs visible
- [ ] Grafana has real metrics (not empty panels)
- [ ] Terminal font size 16px minimum
- [ ] Browser zoom at 100%, no toolbar visible
- [ ] Ambient music at -18 dBFS under narration
- [ ] Narration recorded in one clean take per scene if possible
- [ ] Script read-through timed at 9:00 — add 60s buffer for natural pauses
- [ ] Lower thirds exported as PNG overlays
- [ ] Final export: H.264, 1080p60, under 500 MB

---

*Script authored for VayuGrid / Team Coder4not4 / AI4India Hackathon 2026*
