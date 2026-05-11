import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import poisson
from tqdm import tqdm
from generate_patient import gen_patient
from agentic import agentic_decision

# ---------------------------------------------------------------------------
# Mass Casualty Event definitions  (defined early so argparse can list them)
# ---------------------------------------------------------------------------

MCE_TYPES = [
    {
        "name": "Mass Shooting", "min_pts": 5, "max_pts": 15,
        "ailments": {
            "Ballistic injury (gunshot)":       5,
            "Vascular trauma":                  2,
            "Thoracic trauma":                  2,
            "Abdominal trauma":                 2,
            "Traumatic brain injury (TBI)":     1,
        },
    },
    {
        "name": "Train Derailment", "min_pts": 10, "max_pts": 30,
        "ailments": {
            "Crush injury and crush syndrome":  3,
            "Pelvic and long bone injuries":    3,
            "Traumatic brain injury (TBI)":     3,
            "Thoracic trauma":                  2,
            "Abdominal trauma":                 1,
            "Vascular trauma":                  1,
        },
    },
    {
        "name": "Earthquake", "min_pts": 15, "max_pts": 40,
        "ailments": {
            "Crush injury and crush syndrome":  4,
            "Pelvic and long bone injuries":    3,
            "Traumatic brain injury (TBI)":     2,
            "Thoracic trauma":                  1,
            "Burn injury":                      1,
        },
    },
    {
        "name": "Tornado", "min_pts": 5, "max_pts": 20,
        "ailments": {
            "Traumatic brain injury (TBI)":     3,
            "Pelvic and long bone injuries":    3,
            "Vascular trauma":                  2,
            "Thoracic trauma":                  1,
            "Blast injury":                     1,
        },
    },
    {
        "name": "Tsunami", "min_pts": 10, "max_pts": 25,
        "ailments": {
            "Traumatic brain injury (TBI)":     3,
            "Thoracic trauma":                  3,
            "Crush injury and crush syndrome":  2,
            "Pelvic and long bone injuries":    2,
            "Abdominal trauma":                 1,
        },
    },
    {
        "name": "Building Collapse", "min_pts": 8, "max_pts": 25,
        "ailments": {
            "Crush injury and crush syndrome":  5,
            "Pelvic and long bone injuries":    4,
            "Traumatic brain injury (TBI)":     3,
            "Thoracic trauma":                  2,
            "Abdominal trauma":                 1,
            "Vascular trauma":                  1,
        },
    },
    {
        "name": "Large Fire", "min_pts": 5, "max_pts": 20,
        "ailments": {
            "Burn injury":                      5,
            "Primary blast lung injury (PBLI)": 3,
            "Acute acoustic trauma (AAT)":      1,
        },
    },
    {
        "name": "Public Bomb", "min_pts": 20, "max_pts": 50,
        "ailments": {
            "Blast injury":                     4,
            "Primary blast lung injury (PBLI)": 3,
            "Ballistic injury (gunshot)":       2,
            "Burn injury":                      2,
            "Acute acoustic trauma (AAT)":      2,
            "Traumatic brain injury (TBI)":     2,
            "Vascular trauma":                  1,
        },
    },
    {
        "name": "Chemical Plant Blast", "min_pts": 8, "max_pts": 20,
        "ailments": {
            "CBRN nerve agent exposure":        3,
            "Blast injury":                     3,
            "Burn injury":                      2,
            "Primary blast lung injury (PBLI)": 2,
            "Acute acoustic trauma (AAT)":      1,
        },
    },
    {
        "name": "Stadium Stampede", "min_pts": 15, "max_pts": 40,
        "ailments": {
            "Crush injury and crush syndrome":  5,
            "Pelvic and long bone injuries":    4,
            "Traumatic brain injury (TBI)":     2,
            "Thoracic trauma":                  2,
            "Abdominal trauma":                 1,
        },
    },
]

_MCE_NAMES = [e["name"] for e in MCE_TYPES]
_MCE_BY_NAME = {e["name"].lower(): e for e in MCE_TYPES}

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------

hours = 2
simulation_len = hours * 60  # minutes
patients_per_min = poisson.rvs(mu=0.3, size=simulation_len)

tot_doctors = 5
tot_nurses = 10

# ---------------------------------------------------------------------------
# CLI — force MCE events at specific minutes
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="A&E Agentic Triage Simulation",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=(
        "Examples:\n"
        "  python main.py --mce 30\n"
        "  python main.py --mce 30 'Godzilla Attack'\n"
        "  python main.py --mce 10 Earthquake --mce 45 'Large Fire'\n\n"
        "Available event types:\n"
        + "\n".join(f"  {e['name']} ({e['min_pts']}–{e['max_pts']} casualties)" for e in MCE_TYPES)
    ),
)
parser.add_argument(
    "--mce",
    nargs="+",
    action="append",
    metavar=("MINUTE", "EVENT"),
    help=(
        "Force an MCE at MINUTE (0 to %(max)d). "
        "Optionally name the event type (default: random). "
        "Repeatable." % {"max": simulation_len - 1}
    ),
)
parser.add_argument(
    "--no-random-mce",
    action="store_true",
    help="Disable background random MCE events. Forced --mce events still fire.",
)
args = parser.parse_args()

forced_events = []
if args.mce:
    for tokens in args.mce:
        minute_str = tokens[0]
        event_name_raw = " ".join(tokens[1:]) if len(tokens) > 1 else None

        try:
            minute = int(minute_str)
        except ValueError:
            parser.error(f"MCE minute must be an integer, got: {minute_str!r}")

        if not (0 <= minute < simulation_len):
            parser.error(f"MCE minute {minute} is out of range (0–{simulation_len - 1})")

        if event_name_raw:
            matched = _MCE_BY_NAME.get(event_name_raw.lower())
            if matched is None:
                # Try partial match
                matches = [e for e in MCE_TYPES if event_name_raw.lower() in e["name"].lower()]
                if len(matches) == 1:
                    matched = matches[0]
                elif len(matches) > 1:
                    parser.error(
                        f"Ambiguous event name {event_name_raw!r}. Matches: "
                        + ", ".join(m["name"] for m in matches)
                    )
                else:
                    parser.error(
                        f"Unknown event type {event_name_raw!r}.\n"
                        "Available: " + ", ".join(_MCE_NAMES)
                    )
            event = matched
        else:
            event = random.choice(MCE_TYPES)

        n_pts = random.randint(event["min_pts"], event["max_pts"])
        forced_events.append((minute, event["name"], n_pts, event["ailments"]))

# ---------------------------------------------------------------------------
# Build MCE schedule: random background events + forced events
# ---------------------------------------------------------------------------

# Pre-generate random background MCE schedule (Poisson mean = 1 per 240 min)
mce_schedule = {}
if not args.no_random_mce:
    n_mce = poisson.rvs(mu=simulation_len / 240)
    mce_minutes = sorted(random.sample(range(simulation_len), min(n_mce, simulation_len)))
    for t in mce_minutes:
        event = random.choice(MCE_TYPES)
        n_pts = random.randint(event["min_pts"], event["max_pts"])
        mce_schedule[t] = (event["name"], n_pts, event["ailments"])

# Merge forced events (override any random event at the same minute)
for (minute, name, n_pts, ailments) in forced_events:
    mce_schedule[minute] = (name, n_pts, ailments)

# ---------------------------------------------------------------------------
# Simulation state
# ---------------------------------------------------------------------------

values, counts = np.unique(patients_per_min, return_counts=True)
print("Patient arrival distribution:", values, counts)
print(f"Total routine patients expected: {patients_per_min.sum()}")
print(f"MCE events scheduled: {len(mce_schedule)}")
for t, (name, n, _) in sorted(mce_schedule.items()):
    print(f"  Min {t:4d}: {name} ({n} casualties)")
print()

current_order = []      # sorted list of (esi, clinical_description) for agent context
a_doctors = 5
a_nurses = 10

active_treatments = []  # (finish_minute, staff_type)
patient_log = []        # one dict per patient
waiting_queue = []      # unallocated patients
mce_events_log = []     # (minute, name, n_patients) for plotting

# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

for current_min, patients_this_min in tqdm(
    enumerate(patients_per_min), total=simulation_len, desc="Simulating", unit="min"
):
    # Release staff who finish at or before this minute
    still_busy = []
    for (finish_min, staff_type) in active_treatments:
        if finish_min <= current_min:
            if staff_type == "doctor":
                a_doctors = min(a_doctors + 1, tot_doctors)
            else:
                a_nurses = min(a_nurses + 1, tot_nurses)
        else:
            still_busy.append((finish_min, staff_type))
    active_treatments = still_busy

    # --- Mass casualty event ---
    if current_min in mce_schedule:
        event_name, n_mce_pts, mce_ailments = mce_schedule[current_min]
        tqdm.write(
            f"\n{'!'*60}\n"
            f"  MASS CASUALTY EVENT at min {current_min}: {event_name}\n"
            f"  {n_mce_pts} casualties incoming\n"
            f"{'!'*60}"
        )
        mce_events_log.append((current_min, event_name, n_mce_pts))

        for _ in range(n_mce_pts):
            patient_prompt, ailment = gen_patient(ailment_pool=mce_ailments)
            tqdm.write(f"\n[Min {current_min}] MCE casualty — {ailment['ailment']}")

            old_descs = {t[1] for t in current_order}
            new_order, needs = agentic_decision(
                current_order, patient_prompt,
                tot_doctors=tot_doctors, a_doctors=a_doctors,
                tot_nurses=tot_nurses, a_nurses=a_nurses,
            )
            current_order = new_order

            new_entries = [t for t in current_order if t[1] not in old_descs]
            esi = new_entries[0][0] if new_entries else 1
            agent_desc = new_entries[0][1] if new_entries else patient_prompt[:80]

            record = {
                "arrival_min": current_min,
                "esi": esi,
                "needs": needs,
                "ailment": ailment["ailment"],
                "treatment_minutes": ailment.get("median_treatment_minutes", 60),
                "agent_desc": agent_desc,
                "is_mce": True,
                "mce_event": event_name,
                "service_start": None,
                "wait_time": None,
            }
            patient_log.append(record)
            waiting_queue.append(record)

    # --- Routine patients ---
    for _ in range(patients_this_min):
        patient_prompt, ailment = gen_patient()
        tqdm.write(f"\n[Min {current_min}] New patient — {ailment['ailment']}")

        old_descs = {t[1] for t in current_order}
        new_order, needs = agentic_decision(
            current_order, patient_prompt,
            tot_doctors=tot_doctors, a_doctors=a_doctors,
            tot_nurses=tot_nurses, a_nurses=a_nurses,
        )
        current_order = new_order

        new_entries = [t for t in current_order if t[1] not in old_descs]
        esi = new_entries[0][0] if new_entries else 3
        agent_desc = new_entries[0][1] if new_entries else patient_prompt[:80]

        record = {
            "arrival_min": current_min,
            "esi": esi,
            "needs": needs,
            "ailment": ailment["ailment"],
            "treatment_minutes": ailment.get("median_treatment_minutes", 30),
            "agent_desc": agent_desc,
            "is_mce": False,
            "mce_event": None,
            "service_start": None,
            "wait_time": None,
        }
        patient_log.append(record)
        waiting_queue.append(record)

    # --- Allocate waiting patients to available staff (highest priority first) ---
    waiting_queue.sort(key=lambda p: (p["esi"], p["arrival_min"]))

    still_waiting = []
    for patient in waiting_queue:
        needs = patient["needs"]
        allocated = False
        staff_used = None

        if needs == "doctor":
            if a_doctors > 0:
                a_doctors -= 1
                allocated = True
                staff_used = "doctor"
        else:
            if a_nurses > 0:
                a_nurses -= 1
                allocated = True
                staff_used = "nurse"
            elif a_doctors > 0:
                a_doctors -= 1
                allocated = True
                staff_used = "doctor"

        if allocated:
            patient["service_start"] = current_min
            patient["wait_time"] = current_min - patient["arrival_min"]
            finish = current_min + patient["treatment_minutes"]
            active_treatments.append((finish, staff_used))

            for i, (e, d) in enumerate(current_order):
                if d == patient["agent_desc"]:
                    current_order.pop(i)
                    break
        else:
            still_waiting.append(patient)

    waiting_queue = still_waiting

# Patients still in queue at end: wait = time remaining from arrival
for patient in waiting_queue:
    patient["wait_time"] = simulation_len - patient["arrival_min"]

# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

all_waits = np.array([p["wait_time"] for p in patient_log])
treated = [p for p in patient_log if p["service_start"] is not None]
mce_patients = [p for p in patient_log if p["is_mce"]]

print(f"\n{'='*55}")
print(f"SIMULATION SUMMARY  ({hours}h | {simulation_len} minutes)")
print(f"{'='*55}")
print(f"Routine patients arrived : {len(patient_log) - len(mce_patients)}")
print(f"MCE casualties           : {len(mce_patients)}")
print(f"Total patients           : {len(patient_log)}")
print(f"Patients treated         : {len(treated)}")
print(f"Still waiting at end     : {len(waiting_queue)}")

print(f"\nWait time distribution (minutes) — all patients:")
print(f"  Min:    {all_waits.min():.1f}")
print(f"  P25:    {np.percentile(all_waits, 25):.1f}")
print(f"  Median: {np.median(all_waits):.1f}")
print(f"  Mean:   {all_waits.mean():.1f}")
print(f"  P75:    {np.percentile(all_waits, 75):.1f}")
print(f"  Max:    {all_waits.max():.1f}")

if treated:
    tw = np.array([p["wait_time"] for p in treated])
    print(f"\nWait time for treated patients:")
    print(f"  Median: {np.median(tw):.1f} min  |  Mean: {tw.mean():.1f} min")

esi_groups = {}
for p in patient_log:
    esi_groups.setdefault(p["esi"], []).append(p["wait_time"])
print(f"\nWait time by ESI level:")
for esi in sorted(esi_groups):
    w = np.array(esi_groups[esi])
    print(f"  ESI {esi}: n={len(w):3d} | median {np.median(w):.1f} min | max {w.max():.1f} min")

# ---------------------------------------------------------------------------
# Matplotlib plot
# ---------------------------------------------------------------------------

ESI_COLORS = {1: "#d62728", 2: "#ff7f0e", 3: "#e8c000", 4: "#2ca02c", 5: "#1f77b4"}
ESI_LABELS = {
    1: "ESI 1 — Immediate",
    2: "ESI 2 — Emergent",
    3: "ESI 3 — Urgent",
    4: "ESI 4 — Less Urgent",
    5: "ESI 5 — Non-Urgent",
}

fig, ax = plt.subplots(figsize=(16, 7))
fig.patch.set_facecolor("#f8f9fa")
ax.set_facecolor("#f0f2f5")

# Scatter: routine patients (circle) and MCE patients (X)
for esi in range(1, 6):
    routine = [(p["arrival_min"], p["wait_time"]) for p in patient_log
               if p["esi"] == esi and not p["is_mce"]]
    if routine:
        xs, ys = zip(*routine)
        ax.scatter(xs, ys, c=ESI_COLORS[esi], label=ESI_LABELS[esi],
                   s=55, alpha=0.75, zorder=3, linewidths=0)

    mce_pts = [(p["arrival_min"], p["wait_time"]) for p in patient_log
               if p["esi"] == esi and p["is_mce"]]
    if mce_pts:
        xs, ys = zip(*mce_pts)
        ax.scatter(xs, ys, c=ESI_COLORS[esi], marker="X", s=130,
                   alpha=0.95, zorder=4, edgecolors="black", linewidths=0.6)

# Rolling mean over a 5-patient window
if len(patient_log) >= 5:
    pts_sorted = sorted(patient_log, key=lambda p: p["arrival_min"])
    arr_x = np.array([p["arrival_min"] for p in pts_sorted])
    arr_y = np.array([p["wait_time"]   for p in pts_sorted], dtype=float)
    window = max(5, len(patient_log) // 8)
    kernel = np.ones(window) / window
    roll_y = np.convolve(arr_y, kernel, mode="valid")
    roll_x = arr_x[window // 2: window // 2 + len(roll_y)]
    ax.plot(roll_x, roll_y, color="black", linewidth=2.2, zorder=5,
            label=f"Rolling mean ({window}-pt window)")

# MCE event vertical lines + labels
y_top = ax.get_ylim()[1] if patient_log else 60
for (t, name, n) in mce_events_log:
    ax.axvline(x=t, color="#333333", linestyle="--", linewidth=1.4,
               alpha=0.85, zorder=2)
    ax.text(t + 0.4, y_top * 0.98,
            f"{name}\n({n} pts)",
            rotation=90, va="top", ha="left", fontsize=7.5,
            color="#111111",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#aaaaaa", alpha=0.85))

# Legend: add MCE-patient marker entry
mce_handle = ax.scatter([], [], c="grey", marker="X", s=130,
                        edgecolors="black", linewidths=0.6, label="MCE casualty")
handles, labels_leg = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels_leg,
          loc="upper left", fontsize=9, framealpha=0.9)

mce_count = len(mce_events_log)
ax.set_xlabel("Simulation time (minutes)", fontsize=13)
ax.set_ylabel("Patient wait time (minutes)", fontsize=13)
ax.set_title(
    f"A&E Waiting Times  —  {hours}h simulation  |  "
    f"{len(patient_log)} patients  |  {mce_count} mass casualty event{'s' if mce_count != 1 else ''}",
    fontsize=13,
)
ax.grid(True, alpha=0.35, linestyle="--")
ax.set_xlim(-1, simulation_len + 1)

plt.tight_layout()
plt.savefig("waiting_times.png", dpi=150, bbox_inches="tight")
tqdm.write("\nPlot saved to waiting_times.png")
plt.show()
