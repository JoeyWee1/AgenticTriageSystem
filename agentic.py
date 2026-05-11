import json
import re
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

_llm = None

def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=600)
    return _llm


def _parse_json(text):
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
    return {}


def _severity_agent(patient_prompt, current_order):
    queue_str = "\n".join(
        f"  ESI {t[0]}: {t[1][:80]}" for t in current_order
    ) or "  (empty)"

    system = (
        "You are a triage severity assessment agent for an NHS A&E department. "
        "Assess the patient and output ONLY valid JSON with these exact fields:\n"
        '{"esi": <int 1-5>, "description": "<one sentence clinical summary>", '
        '"needs": "<doctor or nurse>", "language_warning": <null or "Warning: input in <language>">}\n'
        "ESI: 1=immediate life threat, 2=emergent, 3=urgent, 4=less urgent, 5=non-urgent.\n"
        "Nurses handle ESI 4-5 non-complex cases; doctors for ESI 1-3 or complex cases."
    )
    user = f"Current queue:\n{queue_str}\n\nNew patient:\n{patient_prompt}\n\nOutput JSON only."

    result = _parse_json(_get_llm().invoke([SystemMessage(content=system), HumanMessage(content=user)]).content)

    esi = max(1, min(5, int(result.get("esi", 3))))
    description = result.get("description", patient_prompt.split("\n")[0])
    needs = result.get("needs", "doctor")
    if needs not in ("doctor", "nurse"):
        needs = "doctor"
    warning = result.get("language_warning")
    if warning:
        print(f"  [Severity Agent] {warning}")

    return esi, description, needs


def _time_agent(esi, description):
    system = (
        "You are a treatment time estimation agent. "
        "Output ONLY valid JSON: {\"treatment_minutes\": <float>}\n"
        "Reference times by ESI: 1→180 min, 2→90 min, 3→45 min, 4→20 min, 5→10 min."
    )
    user = f"ESI: {esi}\nCondition: {description}\nOutput JSON only."

    result = _parse_json(_get_llm().invoke([SystemMessage(content=system), HumanMessage(content=user)]).content)
    return float(result.get("treatment_minutes", 30.0))


def _capacity_agent(queue_len, treatment_minutes, tot_doctors, a_doctors, tot_nurses, a_nurses):
    total_beds = 50
    used_beds = min(
        queue_len + (tot_doctors - a_doctors) + (tot_nurses - a_nurses),
        total_beds,
    )
    bor = (used_beds / total_beds) * 100

    total_staff = tot_doctors + tot_nurses
    active_staff = (tot_doctors - a_doctors) + (tot_nurses - a_nurses)
    sor = (active_staff / total_staff * 100) if total_staff > 0 else 0.0

    buffer_variance = (total_beds * 0.85) - used_beds
    post_bor = ((used_beds + 1) / total_beds) * 100

    flags = []
    if bor > 92:
        flags.append("CRITICAL: BOR > 92%")
    elif bor > 85:
        flags.append("High Gridlock Risk")

    if bor < 85:
        flow = "Stable"
    elif bor < 92:
        flow = "Under Pressure"
    else:
        flow = "Critical"

    return {
        "bor": round(bor, 1),
        "sor": round(sor, 1),
        "post_bor": round(post_bor, 1),
        "buffer_variance": round(buffer_variance, 1),
        "flow_status": flow,
        "flags": flags,
        "can_absorb_cat1": buffer_variance > 0,
    }


def _triage_agent(esi, description, capacity):
    if esi <= 2:
        track = "Major (Severe)"
        action = "Immediate assessment — move to Resus bay now"
    elif esi == 3:
        track = "Major (Severe)"
        action = "Priority assessment required within 30 minutes"
    elif capacity["bor"] > 95 and esi >= 4:
        track = "External Referral"
        action = "Refer to local GP Hub or Urgent Care Centre (BOR > 95%)"
    else:
        track = "Minors (Non-Severe)"
        action = "Standard minors track — await next available clinician"

    if not capacity["can_absorb_cat1"] and esi <= 2:
        action += " | *** IMMEDIATE TRANSFER ALERT ***"

    return {"track": track, "action": action}


def _summary_agent(patient_prompt, esi, description, treatment_minutes, capacity, triage):
    system = (
        "Generate two structured A&E triage reports. Output ONLY valid JSON:\n"
        '{"nurse_report": {"patient_condition": "...", "estimated_treatment_time": "...", '
        '"hospital_capacity_status": "...", "triage_rationale": "...", "triage_decision": "..."}, '
        '"patient_report": {"condition_summary": "...", "estimated_wait": "...", "queue_position_note": "..."}}'
    )
    user = (
        f"Patient: {patient_prompt[:250]}\n"
        f"ESI: {esi} | Condition: {description}\n"
        f"Treatment time: {treatment_minutes:.0f} min\n"
        f"Hospital flow: {capacity['flow_status']} (BOR {capacity['bor']}%)\n"
        f"Track: {triage['track']} | Action: {triage['action']}\n"
        f"Output JSON only."
    )
    return _parse_json(_get_llm().invoke([SystemMessage(content=system), HumanMessage(content=user)]).content)


def _review_agent(patient_prompt, esi, triage, capacity):
    system = (
        "You are a clinical review agent assessing triage decisions. "
        "Output ONLY valid JSON: {\"confidence\": <1, 2, or 3>, \"reason\": \"<one sentence>\"}\n"
        "1=Not confident, 2=Moderate, 3=Confident."
    )
    user = (
        f"Patient: {patient_prompt[:200]}\n"
        f"ESI assigned: {esi}\n"
        f"Track: {triage['track']}\n"
        f"Action: {triage['action']}\n"
        f"Hospital flow: {capacity['flow_status']} (BOR {capacity['bor']}%)\n"
        f"Is this triage decision clinically appropriate? Output JSON only."
    )
    result = _parse_json(_get_llm().invoke([SystemMessage(content=system), HumanMessage(content=user)]).content)
    confidence = max(1, min(3, int(result.get("confidence", 2))))
    reason = result.get("reason", "Assessment complete.")
    return confidence, reason


def agentic_decision(current_order, patient_prompt,
                     tot_doctors=5, a_doctors=5, tot_nurses=10, a_nurses=10):
    """Run the full multi-agent triage pipeline for a new patient.

    Args:
        current_order: list of (esi, description) tuples for waiting patients.
        patient_prompt: natural-language description of the new patient.
        tot_doctors, a_doctors, tot_nurses, a_nurses: staffing state.

    Returns:
        (new_order, needs) where new_order is the updated sorted queue and
        needs is "doctor" or "nurse".
    """
    max_retries = 2

    for attempt in range(max_retries + 1):
        esi, description, needs = _severity_agent(patient_prompt, current_order)
        treatment_minutes = _time_agent(esi, description)
        capacity = _capacity_agent(
            len(current_order), treatment_minutes,
            tot_doctors, a_doctors, tot_nurses, a_nurses,
        )
        triage = _triage_agent(esi, description, capacity)
        summary = _summary_agent(patient_prompt, esi, description, treatment_minutes, capacity, triage)
        confidence, reason = _review_agent(patient_prompt, esi, triage, capacity)

        if capacity["flags"]:
            print(f"  [Capacity] {' | '.join(capacity['flags'])}")

        if confidence == 3 or attempt == max_retries:
            if attempt == max_retries and confidence < 3:
                print(f"  [ESCALATE] Human judgement required — {reason}")
            break

        print(f"  [Review loop {attempt + 1}] Confidence {confidence}/3: {reason} — retrying...")

    nurse_report = summary.get("nurse_report", {})
    patient_report = summary.get("patient_report", {})
    if nurse_report:
        print(f"  [Triage] ESI {esi} | {triage['track']} | {triage['action']}")
    if patient_report:
        cond = patient_report.get("condition_summary", "")
        wait = patient_report.get("estimated_wait", "")
        if cond:
            print(f"  [Patient report] {cond} | Est. wait: {wait}")

    new_patient = (esi, description)
    new_order = sorted(list(current_order) + [new_patient], key=lambda x: x[0])

    return new_order, needs
