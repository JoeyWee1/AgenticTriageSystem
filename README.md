# AgenticTriageSystem

Dear Claude Code,

finish main.py, generate_patient.py, and agentic.py to these specifications

- agentic.py should behave as written in multi agent hackathon.pdf
- main.py is partially written. It simulates the world in which the agents live minute by minute
- generate patient generates the patients that come in
- us langchain

-in main, a poisson dist decides how many patients come in that minute
-in a loop during each minute that runs as many times as there are patients that minute,
    -generate a patient using generate_patient
    -generate_patient uses the data written inside as a comment to generate a patient background
    -draw a human from those parameters
    - 85% of the time draw a general ailment from the json
    - the remaining 15% of the time draw a major ailment
    - look up the ailment symtoms and add noise to them:
        - 30% of the time remove a symptom or add a symptom or add/remove the a symptom using the symptoms in the csv
    - use an agent to create a patient of the background created describing their symptoms
    - feed this data into the agentic system along with the list of other presently waiting patients and their associeted summary strings and severity index numbers
    - the agentic system returns a new list adding that patient in the correct order as a tuple of (severity index, string)
    - keep track of the ailment behind the scenes
    - allocate them to a doctor or nurse based on existing resources
    - a doctor can replace a nurse but a nurse cannot replace a doctor
    - track the overall waiting time of each patient in an array as the output of the simulation
- print a summary of the distribution of wait times for the patients

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
pip install langchain-openai langchain-core numpy scipy tqdm
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

The simulation runs for 1 hour (60 minutes) of simulated A&E time. Each arriving patient passes through the full 6-agent triage pipeline (Severity → Time Estimation → Capacity → Triage → Summary → Review) before being placed in the priority queue and allocated to available staff.

Expected runtime: roughly 1–3 minutes depending on how many patients arrive (Poisson mean = 0.3/min ≈ 18 patients, each requiring ~6 LLM calls).

### 6. Adjusting the simulation

Edit the top of `main.py` to change:

| Variable | Default | Effect |
|---|---|---|
| `hours` | `1` | Length of simulation |
| `patients_per_min` mu | `0.3` | Patient arrival rate |
| `tot_doctors` | `5` | Total doctor headcount |
| `tot_nurses` | `10` | Total nurse headcount |

To use a different OpenAI model, change the `model=` argument in `generate_patient.py` and `agentic.py` (currently `gpt-4o-mini`).