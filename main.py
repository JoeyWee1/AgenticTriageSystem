import numpy as np
from scipy.stats import poisson
from generate_patient import gen_patient
from agentic import agentic_decision

hours = 1
simulation_len = hours * 60  # in minutes
patients_per_min = poisson.rvs(mu=0.3, size=simulation_len)  # approx 0.3 patients per minute at Charing Cross Hospital

tot_doctors = 5
tot_nurses = 10

values, counts = np.unique(patients_per_min, return_counts=True)
print("Patient arrival distribution:", values, counts)
print(f"Total patients expected: {patients_per_min.sum()}\n")

current_order = []  # sorted list of (esi, clinical_description) for agent context
a_doctors = 5       # available doctors
a_nurses = 10       # available nurses

# (finish_minute, staff_type) for all active treatments
active_treatments = []

# Full patient records: one dict per patient
patient_log = []

# Unallocated patient queue: list of dicts with arrival info
waiting_queue = []

for current_min, patients_this_min in enumerate(patients_per_min):
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

    # Generate and triage new patients arriving this minute
    for _ in range(patients_this_min):
        patient_prompt, ailment = gen_patient()
        print(f"\n[Min {current_min}] New patient — {ailment['ailment']}")

        old_descs = {t[1] for t in current_order}

        new_order, needs = agentic_decision(
            current_order, patient_prompt,
            tot_doctors=tot_doctors, a_doctors=a_doctors,
            tot_nurses=tot_nurses, a_nurses=a_nurses,
        )
        current_order = new_order

        # Identify the new patient's ESI and agent description
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
            "service_start": None,
            "wait_time": None,
        }
        patient_log.append(record)
        waiting_queue.append(record)

    # Allocate waiting patients to available staff (highest priority first)
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
        else:  # nurse case
            if a_nurses > 0:
                a_nurses -= 1
                allocated = True
                staff_used = "nurse"
            elif a_doctors > 0:  # doctor can cover a nurse case
                a_doctors -= 1
                allocated = True
                staff_used = "doctor"

        if allocated:
            patient["service_start"] = current_min
            patient["wait_time"] = current_min - patient["arrival_min"]
            finish = current_min + patient["treatment_minutes"]
            active_treatments.append((finish, staff_used))

            # Remove from agent queue context
            for i, (e, d) in enumerate(current_order):
                if d == patient["agent_desc"]:
                    current_order.pop(i)
                    break
        else:
            still_waiting.append(patient)

    waiting_queue = still_waiting

# Patients still waiting when simulation ends
for patient in waiting_queue:
    patient["wait_time"] = simulation_len - patient["arrival_min"]

# Summary statistics
all_waits = np.array([p["wait_time"] for p in patient_log])
treated = [p for p in patient_log if p["service_start"] is not None]

print(f"\n{'='*50}")
print(f"SIMULATION SUMMARY ({hours}h, {simulation_len} minutes)")
print(f"{'='*50}")
print(f"Total patients arrived:   {len(patient_log)}")
print(f"Patients treated:         {len(treated)}")
print(f"Still waiting at end:     {len(waiting_queue)}")
print(f"\nWait time distribution (minutes):")
print(f"  Min:    {all_waits.min():.1f}")
print(f"  P25:    {np.percentile(all_waits, 25):.1f}")
print(f"  Median: {np.median(all_waits):.1f}")
print(f"  Mean:   {all_waits.mean():.1f}")
print(f"  P75:    {np.percentile(all_waits, 75):.1f}")
print(f"  Max:    {all_waits.max():.1f}")

if treated:
    treated_waits = np.array([p["wait_time"] for p in treated])
    print(f"\nWait time for treated patients:")
    print(f"  Median: {np.median(treated_waits):.1f} min")
    print(f"  Mean:   {treated_waits.mean():.1f} min")

esi_groups = {}
for p in patient_log:
    esi_groups.setdefault(p["esi"], []).append(p["wait_time"])

print(f"\nWait time by ESI level:")
for esi in sorted(esi_groups):
    waits = np.array(esi_groups[esi])
    print(f"  ESI {esi}: n={len(waits):3d} | median {np.median(waits):.1f} min | max {waits.max():.1f} min")
