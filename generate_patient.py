# Patient profile ranges
#Patient profile:
# Sex: [1 = Female, 2 = Male]
# ethnicity (in percentage): [Asian = 20.7	Black = 13.5	 Mixed = 5.7	White British = 36.8	  White other = 17.0	 other = 6.3]
# Age (years): [16 - 96]
# Arrival mode: [1 = Walking, 2 = Public Ambulance, 3 = Private Vehicle, 4 = Private Ambulance, 5 = Other]
# Mental: [1 = Alert, 2 = Verbal Response, 3 = Pain Response, 4 = Unresponsive]
# Pain: [1=yes, 0 = no]
# Patient_pain_level (patient assessment of pain for the patient) : [1-10 with 10 as most severe]
# SBP (Systolic Blood Pressure): [ Max=  275.0 | Min=    50.0 ]
# DBP (Diastolic Blood Pressure): [ Max=   160.0 | Min=  31.0 ]
# HR (Heat Rate): [Max=   148.0 | Min=    32.0]
# RR (Respiratory rate): [Max=    30.0 | Min=    14.0]
# BT (Body Temperature): [Max=    41.0 | Min =   35.0]

import json
import random
import csv
import os
import numpy as np
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

_DIR = os.path.dirname(__file__)

with open(os.path.join(_DIR, "data", "general_ailments.json")) as f:
    _GENERAL = json.load(f)

with open(os.path.join(_DIR, "data", "major_incident_ailments.json")) as f:
    _MAJOR = json.load(f)

_EXTRA_SYMPTOMS = []
with open(os.path.join(_DIR, "data", "symptoms.csv")) as f:
    reader = csv.DictReader(f)
    for row in reader:
        _EXTRA_SYMPTOMS.append(row["symptom_name"])

_llm = None

def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=300)
    return _llm

_NAMES_BY_ETHNICITY = {
    "Asian":         (["Priya", "Arjun", "Mohammed", "Fatima", "Aisha", "Raj", "Sunita", "Ravi"],
                      ["Patel", "Khan", "Singh", "Ahmed", "Sharma", "Ali", "Chaudhry", "Begum"]),
    "Black":         (["Kwame", "Aisha", "Kofi", "Abena", "David", "Grace", "Michael", "Yemi"],
                      ["Johnson", "Williams", "Brown", "Davis", "Okafor", "Mensah", "Adeyemi", "Diallo"]),
    "Mixed":         (["Jordan", "Riley", "Morgan", "Taylor", "Alex", "Sam", "Jamie", "Kai"],
                      ["Smith", "Jones", "Williams", "Brown", "Taylor", "Davies", "Wilson", "Moore"]),
    "White British": (["James", "Emma", "Thomas", "Lucy", "Robert", "Sarah", "John", "Chloe"],
                      ["Smith", "Jones", "Williams", "Brown", "Taylor", "Davies", "Wilson", "Moore"]),
    "White other":   (["Elena", "Stefan", "Marta", "Andrei", "Katarzyna", "Ivan", "Anna", "Dmitri"],
                      ["Novak", "Kowalski", "Muller", "Fischer", "Petrov", "Ionescu", "Horvat", "Weber"]),
    "other":         (["Mei", "Jin", "Yuki", "Carlos", "Ana", "Hassan", "Nadia", "Omar"],
                      ["Chen", "Wang", "Li", "Rodriguez", "Martinez", "Hassan", "Nasser", "Kim"]),
}

_ETHNICITY_WEIGHTS = {
    "Asian": 0.207, "Black": 0.135, "Mixed": 0.057,
    "White British": 0.368, "White other": 0.170, "other": 0.063,
}

_ARRIVAL_MODES = {
    1: "walked in",
    2: "brought by public ambulance",
    3: "arrived by private vehicle",
    4: "arrived by private ambulance",
    5: "brought by other means",
}

_MENTAL_STATUS = {
    1: "alert and oriented",
    2: "responding to verbal stimuli",
    3: "responding only to pain",
    4: "unresponsive",
}

_HISTORIES = [
    "no significant medical history",
    "known hypertension on amlodipine",
    "type 2 diabetes on metformin",
    "asthma managed with salbutamol inhaler",
    "previous myocardial infarction 3 years ago",
    "no regular medications",
    "depression managed with sertraline",
    "hypothyroidism on levothyroxine",
    "chronic kidney disease stage 2",
    "atrial fibrillation on warfarin",
]


def _sample_ethnicity():
    ethnicities = list(_ETHNICITY_WEIGHTS.keys())
    weights = list(_ETHNICITY_WEIGHTS.values())
    return random.choices(ethnicities, weights=weights, k=1)[0]


def _sample_vitals(ailment_severity):
    """Sample vital signs biased by ailment severity."""
    if ailment_severity in ("critical", "serious"):
        sbp = random.randint(50, 160)
        hr = random.randint(100, 148)
        rr = random.randint(20, 30)
        bt = round(random.uniform(37.5, 41.0), 1)
        mental = random.choices([1, 2, 3, 4], weights=[0.2, 0.4, 0.3, 0.1])[0]
        pain_level = random.randint(7, 10)
    elif ailment_severity == "moderate":
        sbp = random.randint(90, 200)
        hr = random.randint(70, 120)
        rr = random.randint(16, 24)
        bt = round(random.uniform(36.5, 39.0), 1)
        mental = random.choices([1, 2, 3, 4], weights=[0.7, 0.2, 0.08, 0.02])[0]
        pain_level = random.randint(4, 8)
    else:  # mild
        sbp = random.randint(100, 160)
        hr = random.randint(60, 100)
        rr = random.randint(14, 20)
        bt = round(random.uniform(36.0, 38.5), 1)
        mental = 1
        pain_level = random.randint(1, 6)

    dbp = max(31, min(160, sbp - random.randint(30, 60)))
    return {
        "sbp": sbp, "dbp": dbp, "hr": hr, "rr": rr, "bt": bt,
        "mental": mental, "pain_level": pain_level,
    }


def gen_patient(major_only=False):
    """Generate a simulated A&E patient.

    Args:
        major_only: if True, always draw from major incident ailments (used for MCE patients).

    Returns:
        (description, ailment_dict) where description is a natural-language
        patient self-report and ailment_dict is the ground-truth ailment record.
    """
    ethnicity = _sample_ethnicity()
    first_names, last_names = _NAMES_BY_ETHNICITY[ethnicity]
    name = f"{random.choice(first_names)} {random.choice(last_names)}"
    age = random.randint(16, 96)
    gender = random.choice(["male", "female"])
    history = random.choice(_HISTORIES)
    arrival_mode = random.choices(list(_ARRIVAL_MODES.keys()), weights=[0.45, 0.30, 0.15, 0.05, 0.05])[0]

    ailment = random.choice(_MAJOR) if (major_only or random.random() >= 0.85) else random.choice(_GENERAL)
    severity = ailment.get("severity", "mild")
    vitals = _sample_vitals(severity)

    symptoms = list(ailment["symptoms"])
    if random.random() < 0.30:
        action = random.choice(["remove", "add", "swap"])
        if action == "remove" and len(symptoms) > 2:
            symptoms.pop(random.randrange(len(symptoms)))
        elif action == "add":
            extra = random.choice(_EXTRA_SYMPTOMS)
            if extra not in symptoms:
                symptoms.append(extra)
        else:  # swap
            if len(symptoms) > 1:
                symptoms.pop(random.randrange(len(symptoms)))
            extra = random.choice(_EXTRA_SYMPTOMS)
            if extra not in symptoms:
                symptoms.append(extra)

    symptom_str = "; ".join(symptoms)
    llm_prompt = (
        f"A {age}-year-old {gender} with {history} presents to A&E. "
        f"They {_ARRIVAL_MODES[arrival_mode]}. "
        f"Their symptoms are: {symptom_str}. "
        f"Write 2-3 sentences in first person as the patient describing how they feel to a nurse. "
        f"Be natural and realistic. Do not name the medical condition."
    )

    response = _get_llm().invoke([HumanMessage(content=llm_prompt)])
    mental_str = _MENTAL_STATUS[vitals["mental"]]
    description = (
        f"Patient: {name} | Age: {age} | Gender: {gender} | Ethnicity: {ethnicity}\n"
        f"History: {history} | Arrival: {_ARRIVAL_MODES[arrival_mode]}\n"
        f"Vitals: BP {vitals['sbp']}/{vitals['dbp']} | HR {vitals['hr']} | "
        f"RR {vitals['rr']} | Temp {vitals['bt']}°C | Mental: {mental_str} | "
        f"Pain: {vitals['pain_level']}/10\n"
        f"In their own words: {response.content.strip()}"
    )

    return description, ailment
