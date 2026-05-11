# AgenticTriageSystem

A multi-agent simulation of an NHS A&E department, built for the Cambridge Infosys Hackathon. The system generates realistic patients, triages them through a 6-agent LLM pipeline, allocates staff, and tracks waiting times — including the impact of mass casualty events.

---

## Architecture

### Patient Generation (`generate_patient.py`)

Each patient is drawn from realistic population statistics for Charing Cross Hospital:

- Demographics sampled from ethnicity-weighted distributions (age 16–96, gender, medical history, arrival mode, vitals)
- **85%** of patients present with a general ailment; **15%** with a major incident ailment
- Symptom noise applied with 30% probability: a symptom is randomly added, removed, or swapped using the symptoms catalogue
- GPT-4o-mini writes a natural first-person description of how the patient feels

### Triage Pipeline (`agentic.py`)

Each patient passes through six agents in sequence, backed by GPT-4o-mini via LangChain:

| # | Agent | Input | Output |
|---|---|---|---|
| 1 | **Severity** | Patient description + current queue | ESI score (1–5), clinical summary, doctor/nurse needed, language warning |
| 2 | **Time Estimation** | ESI + condition | Estimated treatment time (minutes) |
| 3 | **Capacity** | Live staff counts + queue length | BOR, SOR, buffer variance, flow status |
| 4 | **Triage** | ESI + capacity data | Track (Severe / Non-Severe / Referral), action plan |
| 5 | **Summary** | All of the above | Nurse report + patient-facing report |
| 6 | **Review** | All of the above | Confidence score 1–3, one-sentence rationale |

If the Review Agent scores below 3, the pipeline loops back and retries (max 2 retries). If confidence is still below 3 after retries, the case is escalated for human judgement.

The Capacity Agent uses live simulation state (available doctors, nurses, queue length) passed directly from the environment each minute.

### Simulation Loop (`main.py`)

- Runs minute by minute for a configurable number of hours
- Patient arrivals follow a Poisson process (mean 0.3/min)
- Staff are allocated from the front of the priority queue (ESI 1 first)
- Doctors can cover nurse-level cases; nurses cannot cover doctor-level cases
- Treatment durations are drawn from ailment data; staff become available again when treatment ends
- At the end, a `waiting_times.png` plot is produced showing wait time per patient over time, coloured by ESI, with mass casualty events marked

---

## How to Run

### 1. Prerequisites

- Python 3.9+
- An OpenAI API key

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install langchain-openai langchain-core numpy scipy tqdm matplotlib
```

### 4. Set your OpenAI API key

**Windows (PowerShell)**
```powershell
$env:OPENAI_API_KEY = "sk-..."
```

**macOS / Linux**
```bash
export OPENAI_API_KEY="sk-..."
```

### 5. Run the simulation

```bash
python main.py
```

Each patient requires ~6 LLM calls. Expected wall-clock runtime scales with total patient count (routine + MCE casualties).

---

## Mass Casualty Events (MCE)

Mass casualty events fire randomly in the background at an average rate of one per 240 simulated minutes (Poisson process). When an event fires, a batch of casualties arrives simultaneously — all drawn from major incident ailments — and each passes through the full 6-agent triage pipeline.

### Forcing events at specific minutes

```bash
# Random event type at minute 30
python main.py --mce 30

# Specific event type (exact or partial name, case-insensitive)
python main.py --mce 30 "Public Bomb"
python main.py --mce 30 bomb

# Multiple forced events
python main.py --mce 10 Earthquake --mce 90 "Large Fire"

# Disable background random MCEs (only forced events fire)
python main.py --no-random-mce
python main.py --no-random-mce --mce 30 Earthquake
```

Forced events are merged with any randomly scheduled events. If a forced minute collides with a random one, the forced event takes priority.

Run `python main.py --help` for a full reference.

### Available event types

| Event | Casualties |
|---|---|
| Mass Shooting | 5–15 |
| Train Derailment | 10–30 |
| Earthquake | 15–40 |
| Tornado | 5–20 |
| Tsunami | 10–25 |
| Building Collapse | 8–25 |
| Large Fire | 5–20 |
| Public Bomb | 20–50 |
| Chemical Plant Blast | 8–20 |
| Stadium Stampede | 15–40 |

---

## Configuration

Edit the top of `main.py` to change simulation parameters:

| Variable | Default | Effect |
|---|---|---|
| `hours` | `2` | Length of simulation |
| `patients_per_min` mu | `0.3` | Routine patient arrival rate (per minute) |
| `tot_doctors` | `5` | Total doctor headcount |
| `tot_nurses` | `10` | Total nurse headcount |

To use a different OpenAI model, change the `model=` argument in `generate_patient.py` and `agentic.py` (currently `gpt-4o-mini`).
