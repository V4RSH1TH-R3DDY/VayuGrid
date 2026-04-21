# VayuGrid — Team Implementation Plan
### Who builds what, when, and why
**Team: Coder4not4 | AI4India Hackathon**

---

## Meet the Team & Their Roles

Before diving into phases, here's how the work is divided. Each person owns a clear vertical slice of the system from start to finish — not just "frontend" or "backend," but a complete domain they're responsible for end-to-end.

| Person | Domain | What They Own |
|--------|--------|---------------|
| **Mukul** | Infrastructure & Integration | The foundation everything runs on — monorepo, CI/CD, Docker, databases, deployment, and the glue that connects all services together |
| **Charithra** | AI & Machine Learning | The brains of the system — the RL agent (CortexCore) and the GNN prediction model (VayuGNN), plus all training pipelines |
| **Varshith** | Data & Simulation | The virtual world the AI learns in — the grid simulator, dataset pipelines, load profiles, and model validation |
| **Anjali** | Product & Security | What users actually see and interact with — all three dashboards, the API layer, and everything related to privacy and security compliance |

These roles overlap at the interfaces — Charithra needs Varshith's simulator to train on, Anjali needs Mukul's API to build dashboards against. Those handoff points are called out explicitly in each phase.

---

## The Big Picture — What Gets Built in What Order

Think of VayuGrid as three concentric layers being built inside-out:

**First**, everyone builds their foundations independently (Phase 0). This is setup work — no one can start building the real system until this is done.

**Second**, Varshith builds the simulation environment (Phase 1) while Mukul sets up the data infrastructure. The simulator is the single biggest dependency in the project — nothing can be trained until it exists.

**Third**, Charithra trains the AI models (Phases 2 & 3) using Varshith's simulator. These run in parallel with each other.

**Fourth**, Mukul builds the communication and trading layer (Phase 4), while Anjali builds the dashboards and security layer (Phases 5 & 6) in parallel.

**Finally**, the whole team converges for real-data validation, testing, deployment, and scale (Phases 7–10).

---

## Phase 0 — Getting Set Up
**Duration: 1 week | Everyone, led by Mukul**

---

### Mukul — Project Infrastructure

Your job in week one is to make sure every other team member can hit the ground running when they start writing actual code. The decisions you make here will either save or cost the team hours every week.

**Set up the monorepo.** All four team members work in a single Git repository divided into folders — one per service. This means shared code can be shared properly, the CI pipeline runs once for everyone, and you're never debugging "it works on my machine" issues. Use `pnpm` workspaces for the JavaScript dashboard and Python `uv` for everything else.

**Containerize the entire development environment.** Write a `docker-compose.yml` that spins up all the services the team needs locally with a single command. This includes the message bus (RabbitMQ), the database (PostgreSQL with TimescaleDB extension for time-series data), a cache layer (Redis), and local monitoring dashboards (Grafana + Prometheus). Every developer should be able to run `docker-compose up` and have a fully working local environment in under 5 minutes.

**Set up the CI/CD pipeline on GitHub Actions.** Every time someone pushes code, the pipeline should automatically check code formatting, run type checks, run unit tests, and build Docker images. When code merges to main, it should additionally run model evaluation benchmarks and deploy to a staging environment. This prevents broken code from ever reaching the team.

**Set up MLflow** for tracking AI experiments. Every time Charithra runs a training job, MLflow automatically logs what hyperparameters were used, what metrics resulted, and saves the model checkpoint. This is how you avoid the situation where "that really good model from two weeks ago" is impossible to reproduce.

---

### Charithra — AI Environment Setup

Your setup work is mostly reading and planning, which will save you weeks of misdirection later.

**Study the three data models that everything in the system is built around.** There are four core data structures: `NodeState` (what a single household looks like at any moment — battery level, solar output, what the EV needs), `TradeOrder` (a record of one energy trade between two houses), `GridTelemetry` (raw readings from distribution transformers), and `NeighborhoodSignal` (a broadcast command from the GNN to all nodes — things like "throttle your load" or "go to island mode"). You will be consuming and producing all of these constantly. Understand them deeply.

**Make the critical architecture decisions for the AI stack and document them.** The two most important ones: first, the GNN will run on a dedicated neighborhood server, not distributed across individual home devices — this keeps Phase 1 achievable. Second, you'll start with single-agent PPO (one AI per household, making decisions independently) before upgrading to multi-agent RL in Phase 10. MARL adds enormous complexity and you need single-agent working well first.

---

### Varshith — Data Environment Setup

Your setup work is about getting the data sources in order before you need them.

**Get access to and explore the Pecan Street Dataport.** This is the best open-source residential energy dataset available — 1-minute interval readings from hundreds of homes. It's American data and will need significant transformation to represent Indian households (lower total consumption, different peak hours, different load shapes), but it's the foundation of the training data.

**Download NREL NSRDB solar data for Indian cities.** You'll need Bengaluru, Mumbai, Delhi, Chennai, and Hyderabad at minimum. This is the irradiance data that gets converted into solar panel output during simulation. Download it now so it's ready when Phase 1 starts.

**Document the baseline metrics the AI must beat.** Before any model is trained, establish what "good enough" looks like. These become the go/no-go gates for each phase:

| Metric | Baseline (no AI) | VayuGrid Target |
|--------|-----------------|-----------------|
| Solar energy wasted (curtailment) | 28% | Less than 5% |
| Peak demand reduction | 0% | More than 25% |
| Transformer overloads per month | ~15 events | Fewer than 2 |
| Average household electricity cost reduction | 0% | More than 20% |
| Switchover to island mode during outage | Full blackout | Under 2 minutes |
| P2P trade settlement speed | Not applicable | Under 500ms |

---

### Anjali — Product & Compliance Setup

Your setup work is about understanding the users and the legal landscape before you design a single screen.

**Map each user persona to specific features.** You have three very different users. Tony (the grid operations manager) needs deep technical visibility and manual override capability — he's calm under pressure and methodical, he wants to see problems before they happen. Reggie (the tech-savvy homeowner with solar panels and an EV) wants to know exactly how much his investment is earning and whether the AI is making good decisions on his behalf. Luigi (the community organizer for a low-income housing society) needs simplicity above everything else — his residents are worried about whether their lights will stay on and whether rich neighborhoods are getting better service.

**Audit the DPDP Act requirements.** The Digital Personal Data Protection Act 2023 creates specific legal obligations for VayuGrid. Document exactly what data you're collecting about each household, what you're using it for, how long you're keeping it, and what rights residents have. The key obligations you need to design for from the start: residents must be able to export all their data on request, residents must be able to request deletion of their data, data can only be used for the stated purpose (grid balancing, not profiling), and raw readings must be deleted after 90 days. These aren't optional and they're not features you add later — design them in from week one.

---

## Phase 1 — Building the Simulation Environment
**Duration: 2 weeks | Varshith leads, Mukul supports**

---

### Varshith — Grid Simulator

This is the most important deliverable in the entire project. Everything Charithra trains, every claim you make about performance, every demo — all of it runs inside this simulator. It needs to be realistic enough that an AI trained entirely inside it can transfer to real hardware.

**What the simulator models.** A residential neighborhood as an electrical network: homes and batteries are nodes, distribution lines connecting them are edges with real electrical properties (resistance, how much voltage drops over distance). Each home has a configurable mix of features — solar panels, a battery, an EV charger, or none of these. The simulator advances in 1-minute steps and returns the updated state of every node after each step.

**Make it configurable for Indian conditions.** The number of homes (start with 10–50, scale to 500 for stress tests), what fraction have solar (40% default), batteries (30%), or EVs (25%), and critically — a city profile that adjusts the shape of the daily load curve. An Indian urban household has a completely different consumption pattern than an American or European one: the afternoon AC surge when ambient temperature exceeds 35°C, the 7–9 PM cooking and lighting peak, festival load bursts during Diwali, cricket match spikes. Build these in explicitly.

**Build the load profile library.** Take Pecan Street data and apply Indian scaling factors: total daily consumption scaled down to roughly 5–8 kWh per household (about 55% of the US average), solar curves replaced with NREL data for the target Indian city, EV charging rates scaled to Indian vehicle types like the Tata Nexon EV. Then add Gaussian noise and day-of-week variation so the training set has diversity.

**Model the transformer faithfully.** The distribution transformer is the physical component VayuGrid is designed to protect. Model its thermal behavior using the IEEE C57.91 standard — as load increases, the winding temperature rises, and above a certain threshold, every extra degree roughly doubles the aging rate. A transformer running at 105°C winding temperature ages twice as fast as one at 98°C. This thermal model is what makes the Duck Curve problem feel real during training.

**Inject fault scenarios.** The AI needs to have seen every failure mode before it encounters one in the real world. Build a library of scenarios to inject during training: transformer overload (load exceeds 120% of rated capacity), sudden solar dropout from cloud cover, central grid outage triggering islanding, and — critically — a planned maintenance window that looks similar to an outage but should *not* trigger automated trading responses. A model that can't distinguish planned maintenance from a real failure is dangerous to deploy.

**Handoff to Charithra:** When the simulator is ready, write a clear README explaining how to instantiate it, how to configure scenarios, and what each output field means. Charithra should be able to start training within an hour of receiving this.

---

### Mukul — Data Infrastructure

While Varshith builds the simulator, set up the persistent data layer that will store all telemetry, trades, and model artifacts.

**Configure TimescaleDB** as the primary database for time-series telemetry. TimescaleDB is PostgreSQL with a time-series extension — it automatically partitions data by time (called hypertables), which keeps queries fast as data accumulates over months. Set up compression policies so data older than 7 days is automatically compressed (time-series data typically compresses 10–20×). The tables you need at this stage: raw node telemetry (one row per node per minute), transformer readings, trade records, and signal history.

**Set up Redis** as a fast cache for the current state of every node. The matching engine needs to look up current prices and battery states in sub-millisecond time — PostgreSQL is too slow for this. Redis holds the live state, PostgreSQL holds the history.

**Wire up the observability stack.** Grafana and Prometheus should be collecting metrics from every service by the time Phase 2 starts. The metrics that matter most: database query latency, message queue depth (a growing queue means something is falling behind), and storage consumption rate (time-series data accumulates fast).

---

## Phase 2 — Training the Edge AI Agent (CortexCore)
**Duration: 3 weeks | Charithra leads, Varshith supports**

---

### Charithra — The PPO Agent

CortexCore is the AI that runs inside each home's Vayu-Node device. Its job is to autonomously manage the household's energy micro-economy — when to charge or discharge the battery, when to sell excess solar to neighbors, when to buy energy from neighbors, and how to make sure the EV is charged by its deadline. It does all of this without human intervention, every minute of every day.

**Designing what the agent sees.** Think of this like designing the dashboard a human energy manager would look at before making a decision. The agent needs to know its own situation (battery level, how much solar is being generated right now, how much power the house is consuming, current grid voltage), its EV situation (current charge level, target level, and how many hours until the deadline), what the market looks like (what prices neighbors are offering to buy and sell at, how many buyers and sellers are currently active), what time of day it is (encoded mathematically so the agent understands the cyclical nature of daily patterns), any signals the GNN is broadcasting about grid stress, and 15-minute-ahead forecasts for solar generation, household demand, and expected prices.

That last item — the forecasts — deserves special attention. Train a small, separate LSTM model to generate these short-horizon predictions. It doesn't need to be perfect; it just needs to give the agent slightly better information than assuming "the next 15 minutes will look exactly like right now."

**Designing what the agent can do.** The agent controls five things simultaneously: how fast to charge or discharge the battery (a dial from full discharge to full charge), how much power to import from or export to the main grid, how fast to charge the EV, what price to offer when selling energy to neighbors, and the maximum price to pay when buying from neighbors. After the agent makes a decision, apply hard physical constraints — things like "you can't discharge faster than the battery's rated C-rate" are physical laws, not things the agent should have to learn through trial and error.

**Designing the reward function — the most important decision.** The reward function is how you translate "what we want the agent to do" into a number it can optimize. Get this wrong and the agent will find clever loopholes. The agent should be rewarded for: making money through P2P trades, reducing electricity bills by using solar and batteries intelligently, and having the EV charged on time (heavy penalty for missing the deadline — this is a real failure for the homeowner). It should be penalized for: running the battery too low or too high (both degrade cells faster), ignoring grid stress signals from the GNN (poor neighbor behavior), and letting solar generation go to waste. Balance these objectives using weights — the economic reward is most important, EV satisfaction is second, cooperation with the grid is third.

**Use curriculum learning — start easy, get harder.** Don't throw the full environment at the agent from epoch zero. Training Stage 1 (the first 500 epochs): single home, no P2P market, fixed predictable load. Stage 2 (epochs 500–1500): introduce the P2P market and varied load profiles. Stage 3 (epochs 1500–3000): add the multi-node environment and fault injection. Stage 4 (3000+): replay real data from Pecan Street and AIKosh. Each stage teaches the agent a new skill before introducing the next complication.

**The deployed model must be tiny.** The neural network that actually runs on a Raspberry Pi 5 needs to compute in under 5 milliseconds. The way to achieve this: use an asymmetric architecture where the actor (the part that makes decisions, deployed to the device) is a small two-layer network, while the critic (used only during training to evaluate how good decisions were) can be much larger. After training, export the actor to ONNX format and apply INT8 quantization, which cuts the model size roughly in half with minimal accuracy loss.

**Handoff to Varshith:** When ready, provide the trained model checkpoint, the observation normalizer (required for correct behavior), and benchmark numbers showing performance on the baseline scenarios from Phase 1.

---

### Varshith — Training Support & Evaluation

Your role during Phase 2 is supporting Charithra with data and evaluation.

**Create diverse training scenarios.** The simulator needs to generate scenarios that cover the full range of conditions the agent will encounter in the real world. Generate training episodes that span all seasons (winter and summer have very different solar curves), all days of the week, various EV deadline pressures, low/medium/high P2P market activity, and a representative mix of fault scenarios. Store these as a reproducible set so training runs are comparable.

**Add domain randomization.** For each training episode, randomly vary: battery capacity ±20% from nominal (manufacturing tolerances), solar panel efficiency reduced by 0–15% (aging and dust), load profile noise ±10%, grid voltage within a realistic range, and a simulated message delay of 50–500ms on market signals. This range of conditions makes the trained agent robust to real-world variation rather than just performing well on the training distribution.

**Run the baseline comparison.** Implement two simple rule-based controllers — one that just follows a fixed schedule, one that uses Model Predictive Control. When Charithra has a trained model, run all three against the same set of held-out test scenarios and measure the metrics from Phase 0. The PPO agent must beat both baselines by at least 20% on total cost to proceed.

---

## Phase 3 — Training the Neighborhood Brain (VayuGNN)
**Duration: 3 weeks | Charithra leads, Varshith supports | Runs in parallel with Phase 2**

---

### Charithra — The GNN Prediction Model

While CortexCore handles intelligence at the household level, VayuGNN handles intelligence at the neighborhood level. It's the system's collective awareness — a Graph Neural Network that watches every node and transformer simultaneously, predicts problems before they happen, and broadcasts pre-emptive signals to coordinate the neighborhood's behavior.

**Why a GNN specifically.** An electrical distribution network is a graph. Homes are connected to junction points, junction points are connected to transformers, transformers are connected to the utility grid. The relationships between these components matter — a problem at one transformer affects all the homes downstream from it, but not homes upstream. Standard neural networks treat all inputs as independent. A GNN explicitly models these relationships through message passing between connected nodes.

**What the model predicts.** Given the last 12 minutes of readings from every node and transformer (12 snapshots of the neighborhood graph), the model should predict: the probability of transformer overload in each of the next 30 minutes, expected voltage levels over the next 30 minutes, a single risk score for the whole neighborhood (0 = everything fine, 1 = critical), and the expected load curve for the next 24 hours (the duck curve forecast, used for pre-emptive price adjustments).

**The architecture in plain English.** The model processes data in two stages. First, a spatial stage: it looks at one snapshot at a time and learns how to pass information between connected nodes — a transformer seeing high load "tells" its connected homes to expect higher prices, homes with excess solar "tell" their neighbors that supply is available. This uses a Heterogeneous Graph Transformer (HGT) — "heterogeneous" because homes and transformers have different types of features and different roles in the network. Second, a temporal stage: it looks across all 12 snapshots and learns which historical patterns are predictive of near-future problems. For example, it learns that "transformer load rising 2% per minute for 10 consecutive minutes, combined with ambient temperature above 38°C, typically precedes an overload within 20 minutes." This temporal reasoning uses multi-head self-attention, the same mechanism behind large language models.

**The training challenge — rare failures.** Transformer failures are rare events relative to normal operation. If you train with standard loss functions, the model quickly learns to predict "no failure" always and achieves 99.9% accuracy while being completely useless. The fix is Focal Loss, which artificially upweights the rare failure cases so the model can't ignore them. Set alpha=0.75 and gamma=2.0 — these values come from the original Focal Loss paper and work well for heavily imbalanced classification.

**The false positive rate is the most important metric.** A false island signal is not just an incorrect prediction — it physically disconnects the neighborhood from the grid unnecessarily, disrupting everyone's power. This rate must stay below 1%. Monitor it obsessively during training and treat it as a hard constraint, not just one metric among many.

**Translating predictions into actions.** The raw numerical predictions need to become actionable commands. Build a simple threshold-based system: if neighborhood risk exceeds 0.85 or predicted voltage will drop below 0.88 per-unit within 5 minutes, broadcast a critical ISLAND signal. If risk exceeds 0.5, broadcast THROTTLE to the top 20% most flexible loads. If a duck curve ramp rate of more than 2 kW/minute is predicted, broadcast PRE_COOL so homes shift their cooling load forward before the evening peak. If risk drops below 0.1 and the grid has been stable for 5 minutes, broadcast RESUME.

**The fairness algorithm — for Luigi's community.** During islanding with limited battery reserves, the market fails as a resource allocation mechanism. Code a priority queue that overrides market pricing during shortages: medical critical loads (CPAP, oxygen concentrators, refrigerated medication) get served first regardless of payment ability, then refrigeration (food safety, especially critical in Indian summers), then basic lighting (one circuit per home), then communications, then comfort cooling allocated proportionally to whatever's left. This priority order should be configurable by the community.

---

### Varshith — Training Data for the GNN

**Build the graph dataset.** Every training sample for the GNN is a sequence of 12 graph snapshots followed by a 30-minute target window. The tricky part is labeling the targets — you need to look at the simulation output and mark every time step where a transformer overload actually occurred (or would have occurred without intervention). Generate at least 3 months of simulated neighborhood data across different city profiles and fault scenarios.

**Critical: time-based data splitting only.** When splitting into train/validation/test sets, never shuffle across time. If your training data includes readings from Monday and Wednesday and your test data includes Tuesday, the model will "see" both the before and after of the same events, artificially inflating its apparent accuracy. Always use the first 70% of chronological data for training, the next 15% for validation, and the final 15% for testing.

---

## Phase 4 — Building the Mesh Network & P2P Trading Layer
**Duration: 2 weeks | Mukul leads, all others support at interfaces**

---

### Mukul — Communication Infrastructure & Trade Engine

This phase builds the layer that makes VayuGrid more than a collection of individual smart homes — the peer-to-peer communication and trading infrastructure that turns them into a collective.

**Node discovery with libp2p.** Every Vayu-Node is a peer in a gossip network. When a node comes online, it discovers its neighbors automatically using mDNS (a zero-configuration local network discovery protocol — no server required). Each node maintains 6 active connections to nearby peers and passes messages by "gossiping" — when a node receives a trade offer or a state update, it forwards it to its 6 neighbors, who forward it to theirs, until the whole neighborhood has the message within a few seconds. Set the message time-to-live at 3 hops, which is sufficient to reach any node in a neighborhood of up to 200 homes.

**Two types of messages flow through the gossip network.** Trade orders ("I have 2 kWh to sell at ₹8/kWh, offer expires in 60 seconds") and state heartbeats (each node announces its current battery level, generation, and price intentions every 30 seconds so neighbors can make informed bidding decisions).

**The matching engine runs on the neighborhood server.** It implements a Continuous Double Auction — the same mechanism used by most financial exchanges. Sellers post asks (minimum price they'll accept), buyers post bids (maximum price they'll pay). Every 10 seconds, the engine checks if the best bid is at or above the best ask; if so, a trade is made at the midpoint price. Orders expire after 60 seconds if not matched.

Anti-manipulation rules to build in from day one: no single order can exceed 5 kWh (prevents one large battery from dominating the market), the price floor and ceiling are set dynamically by the GNN based on current grid stress (sellers earn more during stress — this incentivizes exporting battery power exactly when the grid needs it most), and any node placing more than 10 orders per minute is automatically rate-limited.

**The pricing regime changes with grid stress.** Under normal conditions, the market sets prices freely. When the GNN reports grid stress above a moderate threshold, a dynamic price cap kicks in — the ceiling rises (making selling more profitable, incentivizing export) while the floor rises slightly (making buying slightly more expensive, incentivizing conservation). When the island signal fires, the market suspends entirely and the Fairness Pool takes over allocation.

**The Aya Wallet Ledger.** Every settled trade needs a tamper-evident record. Build a simple hash-linked ledger — not a blockchain (no mining, no consensus protocol needed for a neighborhood-scale system), just an append-only log where each record includes a cryptographic hash of the previous one. If anyone tampers with a record, the hash chain breaks. Both buyer and seller sign each trade with their Ed25519 keys. The ledger exports a monthly billing summary that can be submitted to the utility company to offset grid import/export bills.

**Handoffs required:** You need Charithra's `NeighborhoodSignal` schema to know what to listen for from the GNN. You need Varshith's simulation environment to test the full trade cycle before anything touches real hardware. You need to give Anjali the API endpoints she'll build dashboards against.

---

## Phase 5 — Security & Privacy
**Duration: 1 week, runs in parallel with Phase 4 | Anjali leads**

---

### Anjali — Security Architecture & DPDP Compliance

Security in an energy system is not optional and it's not a feature you add at the end. A compromised Vayu-Node doesn't just leak data — it can manipulate the physical power supply of an entire neighborhood.

**Encryption everywhere.** All data stored in the database is encrypted with AES-256-GCM (an authenticated encryption mode — it detects tampering, not just unauthorized reads). All communication between services uses TLS 1.3. The connection between each Vayu-Node and the neighborhood server uses mutual TLS, meaning both sides verify each other's identity — a fake node can't just connect and start injecting false data. Manage encryption keys through HashiCorp Vault (self-hosted) with 90-day key rotation.

**Data leaves individual nodes anonymized.** Before any telemetry is transmitted, strip the node's identity and reduce the precision of numerical readings (battery level rounded to 2 decimal places, load demand bucketed into 10 ranges rather than transmitted as an exact float). The GNN gets enough detail to do its job; neighbors learn nothing personal about each other.

**The threat model — think through every realistic attack.** A rogue node trying to join the mesh is countered by Ed25519 identity certificates issued by a neighborhood authority — you can't join without a cert. Price manipulation through wash trades (a node appearing as both buyer and seller) is countered by the matching engine's wash trade detection. Old signed messages being replayed to trigger duplicate trades are countered by a timestamp validation window — any message older than 30 seconds is rejected. Adversarial telemetry designed to corrupt the GNN's predictions is countered by anomaly detection that validates all incoming readings before they reach the model. A denial-of-service attack flooding the matching engine is countered by rate limiting and a circuit breaker that temporarily stops accepting new orders when queue depth exceeds a threshold.

**On each device, an anomaly detector runs alongside the AI agent.** Train a small Isolation Forest model on normal operation data. If the device's own sensor readings become anomalous — wildly out-of-range values, physically impossible state transitions — the device enters Safe Mode and stops making autonomous decisions until a human reviews the situation.

**DPDP Act compliance — specific deliverables.** You need to build and ship these specific features:
- A data export endpoint that lets any homeowner download all data stored about their household as a JSON file
- A data deletion endpoint that schedules removal of all personal data within 72 hours
- A consent flow in the homeowner dashboard explaining exactly what data is collected and why, with explicit opt-in
- A retention policy enforced in the database: raw 1-minute readings auto-delete after 90 days, 15-minute aggregates keep for 3 years, anonymized monthly summaries keep indefinitely
- An automated breach notification system that triggers within 72 hours of any detected security incident

---

## Phase 6 — API Layer & Dashboards
**Duration: 2 weeks, runs in parallel with Phase 4 | Anjali leads**

---

### Anjali — Three Dashboards for Three Very Different People

**The shared API backend.** Build a FastAPI server that exposes the data all three dashboards need. Authentication uses JWTs for web clients and API keys for Vayu-Nodes. Rate limiting protects against overload (100 requests/minute for dashboards, 1000/minute for nodes). A WebSocket endpoint streams live telemetry, trade flow, and GNN predictions to dashboards every 5 seconds without the client needing to poll.

**Tony's Dashboard — the operator view.** Tony is a 50-year-old senior engineer who's calm under pressure and methodical. He needs depth, not simplicity. The centerpiece is a **Grid Health Map** — a live visualization of the neighborhood as a network graph, with nodes color-coded by stress level and transformers shown prominently as the critical components. Hovering over any node shows its current voltage and load. Next to this, a **Duck Curve Tracker** shows actual net load for the day overlaid against the GNN's 24-hour forecast — Tony uses this to understand whether the evening ramp is on track. A **Risk Timeline** panel shows the GNN's transformer overload probability scrolling in real time over the next 30 minutes; this is the early warning system that turns reactive grid management into proactive. A **Signal History** log records every broadcast signal with timestamp, type, which nodes received it, and what percentage actually responded. A **Manual Override Panel** lets Tony issue signals directly, adjust price floors and ceilings, or force a specific section of the neighborhood to island — for when the AI is wrong, and it will sometimes be wrong. Finally, a **KPI summary** at the top: curtailment percentage, peak demand reduction, transformer aging rate, and total P2P volume in kWh and ₹.

**Reggie's Dashboard — the homeowner view.** Reggie is a 35-year-old software architect, analytical and slightly competitive. He wants to know whether his expensive solar installation is paying off. Build this as a **mobile-first Progressive Web App** — Reggie checks this on his phone while commuting. The home screen shows **Today's Energy** as a clean flow diagram: solar generated → consumed at home / sold to neighbors, grid imported / received from neighbors, net bill impact in rupees. Below this, **EV Status** is prominent — a large progress bar showing current charge vs target, time remaining until deadline, cost incurred today, and the AI's charging schedule for the rest of the night (so Reggie can see it's planning to charge between 2–4 AM when prices will be lowest). **Battery Health** shows SoC, estimated backup hours, and health percentage. **Earnings** shows today's P2P revenue compared to what standard net metering would have paid — this number is the reason Reggie bought in. A **Live Market** view shows current neighborhood buy/sell prices with a 1-hour trend.

**Luigi's Dashboard — the community view.** Luigi is a 40-year-old community organizer whose residents have varying levels of technical literacy and genuine financial constraints. This dashboard needs to communicate resilience and fairness without a single piece of jargon. The top of the screen says: **"Your community currently has [X] hours of backup power."** This number updates in real time. Below it, **Community Savings** shows total money saved vs utility rates — today, this month, since installation. During islanding events, a live **Fairness Allocation** view shows how energy is being distributed, with the priority hierarchy visible so residents understand why decisions are being made. A large, accessible **"Flag Critical Load"** button lets residents mark that their household has a medical device that must be prioritized during shortages.

---

## Phase 7 — Bringing In Real Data
**Duration: 2 weeks | Varshith leads, Charithra supports**

---

### Varshith — Data Pipeline

Up to this point, all training has used synthetic data from the simulator. Synthetic data is controllable and fast to generate, but it doesn't have the noise, regional quirks, and unexpected edge cases of real Indian grid behavior. Phase 7 bridges that gap.

**Pecan Street transformation pipeline.** Build an automated pipeline that pulls Pecan Street residential data, applies the Indian scaling factors developed in Phase 1, replaces solar curves with NREL data for the target city, adjusts time zones and peak hours for IST, and outputs clean Parquet files ready for training. The pipeline should be idempotent — running it twice produces the same output.

**NREL solar simulation.** Use PVlib and NREL's SAM (System Advisor Model) to compute per-minute solar AC output for given panel specs and city coordinates. Run this once for each city in the dataset and cache the results — it's computationally slow.

**AIKosh sandbox integration.** This is the most valuable dataset and the hardest to get: localized Indian distribution transformer data accounting for regional grid characteristics — higher ambient temperatures, typical Indian "feeder" load profiles, frequent load shedding patterns. Make a formal request for Secure API access to the AIKosh sandbox. The data you need: 3 years of continuous smart meter readings from at least 5,000 residential units and 50 distribution transformers, at 15-minute intervals, with localized weather metadata (solar irradiance and ambient temperature). This is a big ask, but the technical justification is real: models trained only on US data will fail to account for the voltage fluctuations and load patterns specific to Indian urban distribution grids.

**Retrain both models and verify metrics hold.** After building the real-data pipelines, retrain CortexCore and VayuGNN using the augmented dataset. The metrics from Phase 0's baseline table must still be hit — if performance degrades significantly on real data, investigate before proceeding.

---

### Charithra — Model Fine-tuning

Use the real data Varshith delivers to fine-tune both models with transfer learning. The synthetic-trained models give you a good starting point; fine-tuning on real data closes the remaining gap. Watch particularly for the GNN's false positive rate — real-world noise patterns may cause it to trigger more island events than it should. Retune the signal generation thresholds if necessary.

---

## Phase 8 — Testing Everything
**Duration: 1 week | Everyone, coordinated by Mukul**

---

### Mukul — Integration & Infrastructure Tests

**Integration test: the full trade cycle.** Spin up 5 simulated Vayu-Nodes. One has excess solar and posts a sell order; another has an EV that needs charging and posts a buy order. Verify that a match occurs within 500ms, both nodes update their state correctly, and the ledger records the trade with valid cryptographic signatures. This test should run automatically in CI on every merge to main.

**Integration test: the island sequence.** Simulate a central grid voltage collapse. The GNN should detect the voltage drop within 2 prediction intervals (2 minutes). An ISLAND signal should broadcast within 5 seconds. All nodes should enter island mode within 30 seconds of receiving the signal. The P2P market should suspend, and the Fairness Pool should activate. Verify that no trades are generated during the event.

**Load testing targets.** Use Locust for API load tests. Pass/fail criteria: 500 simultaneous nodes sending telemetry with p99 ingestion latency under 100ms; 1,000 trade orders per minute with p99 settlement under 500ms; GNN inference on a 500-node graph under 2 seconds on the neighborhood server CPU; 100 concurrent WebSocket dashboard connections without visible lag.

**Chaos engineering.** Deliberately break things and verify graceful degradation. Kill the neighborhood server mid-operation — nodes should continue operating autonomously and reconnect automatically when the server recovers. Corrupt 10% of incoming telemetry readings — anomaly detection should reject them without crashing the GNN. Flood the trade queue with 10,000 orders per minute — the circuit breaker should activate cleanly. Simulate an instantaneous grid voltage drop to zero — the island sequence must trigger immediately, not after a delay caused by the change being too fast.

---

### Varshith — Baseline Regression Testing

For every code change that touches the simulator or training pipelines, run the full baseline comparison automatically. If the PPO agent's performance drops below the Phase 0 targets on any metric, flag it as a regression before it merges.

---

### Charithra — Model Behavioral Testing

Beyond metric benchmarks, test specific behaviors manually. Does the agent always meet EV deadlines when given sufficient advance notice? Does it respond correctly to THROTTLE signals within two time steps? Does it maintain the minimum battery buffer during prolonged island events? Does the GNN's false positive rate stay below 1% on the held-out test set?

---

### Anjali — End-to-End Dashboard Testing

Test all three dashboards against the staging environment. Verify that real-time WebSocket updates arrive within the expected latency. Verify that the data export and deletion endpoints actually work and produce correct output. Test the consent flow and make sure it's clear to a non-technical user. Run the dashboard through a basic accessibility check.

---

## Phase 9 — Deployment
**Duration: 1 week | Mukul leads**

---

### Mukul — Hardware & Production Infrastructure

**The Vayu-Node hardware stack (per household):** A Raspberry Pi 5 (8GB RAM) or equivalent ARM single-board computer, a 32GB industrial-grade SD card (rated for continuous write cycles — consumer cards fail within months in this application), a USB3 SSD for local telemetry buffering, Wi-Fi 6 with an optional Ethernet port and 4G LTE modem as fallback connectivity, an RS-485 to Modbus RTU adapter for communicating with the smart meter, a small UPS battery backup (the Vayu-Node must stay powered during a grid outage — that's exactly when it needs to be making island decisions), and a DIN rail mountable, IP54-rated enclosure.

**The neighborhood server (one per 50–200 homes):** An Intel NUC 13 Pro or equivalent mini-PC (i5 processor, 32GB RAM, 1TB NVMe SSD) running the full service stack under Docker Compose. This is a physical box located somewhere with reliable power in the neighborhood — a utility substation, a community building, or a dedicated enclosure.

**OTA model update process.** When Charithra trains a better model, it needs to reach every device safely. The process: training completes and the model is registered in MLflow → CI runs the validation suite against the holdout test → if it passes, the model artifact is signed with an ed25519 key → deploy to 5% of neighborhoods as a canary → monitor for 24 hours → if metrics are stable or improved, roll out to 100% → if any metric degrades by more than 5%, automatic rollback. Use Mender.io for OTA delivery with A/B partition rollback — if an update causes a device to fail to boot, it automatically reverts to the previous working partition.

**Hardware-in-the-loop test before any real deployment.** Run 3 physical Raspberry Pi 5 devices for 72 hours with simulated smart meter data injected via serial port. Measure actual inference latency (must be under 5ms), memory consumption, CPU usage, and thermal behavior under sustained load. Verify OTA updates apply without service interruption.

---

## Phase 10 — Scale
**Duration: 2 weeks | Everyone**

---

### Mukul — Federation Architecture

**Horizontal scaling within a neighborhood.** As node count grows past 200 homes, the stateless services (API gateway, matching engine) scale horizontally behind a load balancer — just add more instances. The bottleneck is the stateful services. Upgrade RabbitMQ to a 3-node cluster with quorum queues for high availability. Enable TimescaleDB compression and continuous aggregates so dashboard queries hit pre-computed summaries rather than scanning raw tables. Route all read-heavy dashboard queries to a read replica, keeping the primary database exclusively for writes.

**Multi-neighborhood federation.** Each neighborhood server exposes a Federation API. A Regional Aggregator sits above multiple neighborhoods, running a simpler GNN over neighborhood-level summaries (not individual home data). When neighborhood A has excess energy and neighborhood B has a deficit, the Regional Aggregator proposes an inter-neighborhood trade. Energy physically flows via the utility grid; settlement flows through the Aya Ledger. The aggregator also provides consolidated data to the utility company for distribution planning.

---

### Charithra — MARL Upgrade & Federated Learning

**Multi-Agent RL upgrade.** Single-agent PPO treats each home's AI independently — it has no way to coordinate with neighbors. The proper formulation for VayuGrid is Multi-Agent Reinforcement Learning (MARL), specifically MAPPO with a centralized critic. During training, the critic sees all nodes' states simultaneously and can evaluate how well the neighborhood is cooperating. But at runtime, each node's actor makes decisions using only its own observations — no real-time communication between agents required. The emergent behaviors that appear in MARL but not single-agent PPO are worth the upgrade: nodes spontaneously coordinate storage charging to flatten the duck curve, high-solar nodes learn to signal surplus before the GNN does, and EV charging shifts collectively to off-peak periods without explicit coordination.

**Federated learning for the GNN.** As the system scales to dozens of neighborhoods, the GNN should learn from all of them — but raw telemetry can't leave each neighborhood under DPDP. The solution is federated learning using the Flower framework. Each neighborhood server trains the GNN locally on its own data, then shares only the gradient updates (not the raw data) with a central aggregation server. The server averages the updates using FedAvg and produces an improved global model that gets pushed back out to all neighborhoods. Raw data never leaves its origin neighborhood.

---

### Varshith — Cross-Neighborhood Data Validation

As new neighborhoods come online, validate that the models transfer correctly. Each new neighborhood has different grid characteristics, different load shapes, different solar penetration. Run the baseline comparison for each new neighborhood within the first week of deployment. Flag neighborhoods where model performance is significantly below target — these need localized fine-tuning.

---

### Anjali — Multi-Neighborhood Dashboard

Extend Tony's operator dashboard with a regional view: a map showing all federated neighborhoods with their current risk levels, inter-neighborhood trade flows, and aggregate KPIs. This becomes the view a utility's distribution planning team uses to understand the cumulative impact of VayuGrid across their service territory.

---

## Full Timeline at a Glance

| Week | Mukul | Charithra | Varshith | Anjali |
|------|-------|-----------|----------|--------|
| 1 | Monorepo, Docker, CI/CD | Study schemas, document AI decisions | Get Pecan Street + NREL data | Persona mapping, DPDP audit |
| 2–3 | Data infrastructure (TimescaleDB, Redis) | Start CortexCore training | Build grid simulator | Start security architecture |
| 4–5 | Continue data infra | Continue CortexCore (curriculum stages) | Simulator QA + load profiles | Security implementation |
| 6 | Start mesh network | Start VayuGNN (runs in parallel) | Training data for GNN | Start API backend |
| 7 | Complete mesh + trade engine | Continue VayuGNN | GNN dataset pipeline | Continue API + dashboards |
| 8–9 | API + service wiring | Fine-tune models | Training support + evaluation | Complete all 3 dashboards |
| 10–11 | Integration testing + chaos | Model behavioral testing | Baseline regression testing | Dashboard E2E testing |
| 12 | HIL test + deployment | Model deployment validation | Real data pipeline | DPDP compliance verification |
| 13–14 | Horizontal scaling + federation | MARL upgrade + federated learning | Cross-neighborhood validation | Regional dashboard |

**Total: ~14 weeks** working in parallel (vs 20 weeks sequential)

---

## The Handoffs That Matter Most

These are the moments where one person's work becomes another person's dependency. Communicate early and often at these points:

**Varshith → Charithra (end of Phase 1):** Simulator is ready. Provide a README, example instantiation code, and a clear description of every field in the output. Charithra should be able to start a training run within an hour.

**Charithra → Mukul (during Phase 4):** Provide the `NeighborhoodSignal` schema and the signal generation thresholds. Mukul needs these to build the correct message routing in RabbitMQ.

**Mukul → Anjali (start of Phase 6):** Document all API endpoints with request/response schemas before Anjali starts building dashboards. A moving API is the most common cause of frontend delays.

**Everyone → Anjali (Phase 8):** Each team member writes integration tests for their own component. Anjali coordinates the end-to-end tests that cross component boundaries.

---

*VayuGrid Team Implementation Plan — Coder4not4*
*AI4India Hackathon | April 2026*
