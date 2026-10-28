"""
AI-Powered Healthcare Chatbot  -  core engine
=============================================
Pipeline for every user message
  1. Safety layer      : emergency / self-harm red-flag detection (rules)
  2. NLU               : intent classification (TF-IDF word+char n-grams -> Logistic Regression)
  3. Entity extraction : symptoms (synonym dictionary + typo correction + negation handling), disease names, BMI values
  4. Dialogue manager  : keeps per-session state, asks smart follow-up questions (information-gain style)
  5. Reasoning         : probabilistic (naive-Bayes style) scoring of 21 conditions from the reported symptoms
  6. Response builder  : explanation, self-care, precautions, specialist, urgency, red flags, disclaimer
Educational project - NOT a medical device; it never replaces a doctor.
"""
import re
import math
import difflib
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union

from knowledge_base import (SYMPTOM_SYNONYMS, IMPLIES, DISEASES, SHORT_TO_DISEASE,
                            EMERGENCY_RULES, INTENT_EXAMPLES)

DISCLAIMER = ("_This chatbot gives general health information only. It is not a medical diagnosis and does not "
              "replace a qualified doctor. If you feel very unwell, please see a doctor._")

HELPLINES = ("**India emergency numbers:** 112 (national emergency) or 108 (ambulance).")

MAX_FOLLOWUPS = 4

# prior "how common is it" weights used by the reasoning step
PRIOR = {"Common Cold": 3, "Influenza (Flu)": 2.5, "Gastroenteritis (Stomach Infection / Food Poisoning)": 2.5,
         "Allergic Rhinitis (Allergy)": 2, "Acid Reflux (GERD)": 2, "Migraine": 1.5, "Sinusitis": 1.5,
         "Urinary Tract Infection (UTI)": 1.5, "Conjunctivitis (Pink Eye)": 1.5, "Hypertension (High Blood Pressure)": 1.2,
         "Anemia (Iron deficiency)": 1.2, "COVID-19": 1.2, "Pneumonia": .8, "Tuberculosis (TB)": .5,
         "Jaundice / Hepatitis": .5, "Chickenpox": .5}

SYMPTOM_SPECIALIST = {
    "fever": "General Physician", "high fever": "General Physician", "cough": "General Physician / Pulmonologist",
    "long cough": "Pulmonologist", "shortness of breath": "Pulmonologist", "wheezing": "Pulmonologist",
    "chest pain": "Cardiologist (urgent evaluation)", "palpitations": "Cardiologist", "headache": "General Physician / Neurologist",
    "sensitivity to light": "Neurologist", "dizziness": "General Physician / Neurologist", "numbness": "Neurologist",
    "rash": "Dermatologist", "itching": "Dermatologist", "blisters": "Dermatologist",
    "red eyes": "Ophthalmologist (eye specialist)", "itchy eyes": "Ophthalmologist", "blurred vision": "Ophthalmologist",
    "eye discharge": "Ophthalmologist", "sore throat": "ENT Specialist", "nasal congestion": "ENT Specialist",
    "runny nose": "ENT Specialist", "facial pain": "ENT Specialist", "abdominal pain": "Gastroenterologist",
    "vomiting": "Gastroenterologist", "diarrhea": "Gastroenterologist", "heartburn": "Gastroenterologist",
    "burning urination": "Urologist / Gynaecologist", "frequent urination": "General Physician / Urologist",
    "joint pain": "Orthopaedic / Rheumatologist", "back pain": "Orthopaedic", "increased thirst": "Endocrinologist / Diabetologist",
    "yellow skin": "Gastroenterologist / Hepatologist", "pale skin": "General Physician / Hematologist",
}

COMMON_START = ["Fever", "Headache", "Cough", "Stomach pain", "Sore throat", "Tiredness"]


# =========================================================================================
# helpers
# =========================================================================================
def normalize(text: str) -> str:
    t = text.lower().replace("’", "'")
    t = re.sub(r"(\w)'(\w)", r"\1\2", t)             # don't -> dont
    t = re.sub(r"\bcan not\b", "cannot", t)
    t = re.sub(r"[^a-z0-9\s.;!?,\-]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def shown(symptoms):
    """symptoms for display: hide 'fever'/'cough' when a more specific one implies it"""
    hide = {imp for s in symptoms for imp in IMPLIES.get(s, [])}
    return [s for s in symptoms if s not in hide]


def join_list(items):
    items = list(items)
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


# =========================================================================================
# Symptom extractor
# =========================================================================================
class SymptomExtractor:
    NEGATORS = r"\b(no|not|without|never|dont|doesnt|didnt|havent|hasnt|hadnt|isnt|arent|wasnt|nor|neither|zero|negative for|free of)\b"
    SPLIT = re.compile(r"[.;!?]|\b(?:but|however|though|although|except)\b|\band (?:i )?(?:also |now |am |have |has |feel |got |had )|\band (?:my|the|his|her|i|it|there|also)\b")

    def __init__(self):
        pairs = []
        for canon, phrases in SYMPTOM_SYNONYMS.items():
            for p in phrases:
                pairs.append((normalize(p), canon))
        pairs.sort(key=lambda x: -len(x[0]))
        self.phrase_to_canon = {}
        for p, c in pairs:
            self.phrase_to_canon.setdefault(p, c)
        self.regex = re.compile(r"\b(?:" + "|".join(re.escape(p) for p, _ in pairs) + r")\b")
        self.vocab = {w for p, _ in pairs for w in p.split() if len(w) >= 4}
        self.neg_re = re.compile(self.NEGATORS)

    def _fix_tokens(self, clause: str) -> str:
        out = []
        for tok in clause.split():
            if tok in self.vocab or len(tok) < 5 or not tok.isalpha():
                out.append(tok); continue
            if tok.endswith("s") and tok[:-1] in self.vocab:
                out.append(tok[:-1]); continue
            m = difflib.get_close_matches(tok, self.vocab, n=1, cutoff=0.85)
            out.append(m[0] if m else tok)
        return " ".join(out)

    def extract(self, text: str):
        """returns (positive symptoms (ordered), negated symptoms)"""
        t = normalize(text)
        t = re.sub(r"\b(red|watery|itchy)(?:,| and)? (red|watery|itchy) eyes\b", r"\1 eyes and \2 eyes", t)
        pos, neg = [], []
        # "i have a cold" -> runny nose / congestion  (but not "cold hands")
        if re.search(r"\b(have|got|caught|having|catch|catching|suffering from) (a |the )?(bad |common |mild )?cold\b(?! (hands|feet|extremities|sweat))", t):
            pos += ["runny nose", "nasal congestion"]
        for clause in self.SPLIT.split(t):
            if not clause or not clause.strip():
                continue
            clause = self._fix_tokens(re.sub(r"[,\-]", " ", clause).strip())
            for m in self.regex.finditer(clause):
                canon = self.phrase_to_canon[m.group(0)]
                before = clause[:m.start()]
                negated = bool(self.neg_re.search(before))
                (neg if negated else pos).append(canon)
        # implications
        for s in list(pos):
            for imp in IMPLIES.get(s, []):
                pos.append(imp)
        pos = list(dict.fromkeys(pos))
        neg = [n for n in dict.fromkeys(neg) if n not in pos]
        return pos, neg


# =========================================================================================
# Intent classifier (the "machine learning" part of the NLU)
# =========================================================================================
class IntentClassifier:
    EXTRA = {"thanks": ["ok", "okay", "fine", "alright", "cool", "great", "got it", "understood", "nice", "good"]}

    def __init__(self):
        X, y = [], []
        for intent, ex in INTENT_EXAMPLES.items():
            for e in ex + self.EXTRA.get(intent, []):
                X.append(normalize(e)); y.append(intent)
        self.model = make_pipeline(
            make_union(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
                       TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)),
            LogisticRegression(C=30, max_iter=2000))
        self.model.fit(X, y)
        self.classes_ = [str(c) for c in self.model.classes_]

    def predict(self, text: str):
        p = self.model.predict_proba([normalize(text)])[0]
        i = int(p.argmax())
        return self.classes_[i], float(p[i])


# =========================================================================================
# The chatbot
# =========================================================================================
class HealthBot:
    def __init__(self):
        self.extractor = SymptomExtractor()
        self.intents = IntentClassifier()
        self.sessions = {}
        # disease alias matcher (longest first)
        aliases = []
        for name, d in DISEASES.items():
            for a in d["aliases"] + [d["short"], name.lower()]:
                aliases.append((normalize(a), name))
        aliases.sort(key=lambda x: -len(x[0]))
        self.alias_map = {}
        for a, n in aliases:
            self.alias_map.setdefault(a, n)
        self.alias_re = re.compile(r"\b(?:" + "|".join(re.escape(a) for a, _ in aliases) + r")\b")
        self.emergency = [(k, re.compile(p)) for k, p in EMERGENCY_RULES]

    # ---------------------------------------------------------------- session state
    def _state(self, sid):
        if sid not in self.sessions:
            self.sessions[sid] = self._new_state()
        return self.sessions[sid]

    @staticmethod
    def _new_state():
        return {"pos": [], "neg": [], "asked": [], "pending": None, "followups": 0,
                "awaiting_bmi": False, "last_disease": None, "last_result": False, "turns": 0}

    def reset(self, sid):
        self.sessions[sid] = self._new_state()

    # ---------------------------------------------------------------- public API
    def respond(self, sid: str, message: str) -> dict:
        st = self._state(sid)
        st["turns"] += 1
        raw = (message or "").strip()
        t = normalize(raw)
        if not t:
            return self._out("I did not receive any message. Tell me how you are feeling, e.g. *\"I have fever and headache\"*.", COMMON_START)

        # 1. safety layer ----------------------------------------------------------------
        for kind, rx in self.emergency:
            if rx.search(t):
                return self._emergency(kind)

        # 2. awaiting BMI numbers --------------------------------------------------------
        bmi = self._parse_bmi(raw)
        if bmi:
            st["awaiting_bmi"] = False
            return self._bmi_reply(*bmi)

        # 3. answer to a pending yes/no follow-up ---------------------------------------
        pos, neg = self.extractor.extract(raw)
        if st["pending"]:
            ans = self._yes_no(t)
            if ans and not pos:
                sym = st["pending"]; st["pending"] = None
                st["asked"].append(sym)
                if ans == "yes":
                    self._add(st, [sym], [])
                elif ans == "no":
                    self._add(st, [], [sym])
                return self._assess(st)
            if pos or neg:                       # user gave new info instead of yes/no
                st["asked"].append(st["pending"]); st["pending"] = None
        if not st["pending"] and not pos and not neg and self._yes_no(t) in ("yes", "no") and len(t.split()) <= 3:
            return self._out("Okay! You can tell me more symptoms, ask about a disease, or choose an option below.",
                             ["Add more symptoms", "Which doctor should I see?", "Health tips", "Start over"])
        intent, conf = self.intents.predict(raw)
        disease = self._find_disease(t)

        # keyword rules that make intent detection more robust
        if disease:
            if re.search(r"\b(prevent|prevention|precaution|precautions|avoid|protect|safe from|care|cure|treat|treatment|remed\w+|recover)\b", t):
                intent = "precautions"
            elif re.search(r"\b(doctor|specialist|consult|department|physician)\b", t):
                intent = "specialist"
            elif re.search(r"\b(what is|what are|tell me about|explain|define|about|symptoms of|causes? of|meaning|information|details|describe)\b", t):
                intent = "disease_info"
            elif re.search(r"\b(medicine|tablet|drug|dose|dosage|syrup)\b", t):
                intent = "medication"
        if re.search(r"\b(bmi|body mass index|am i (over|under)weight)\b", t) or (intent == "bmi" and not pos and not neg):
            return self._bmi_prompt(st)
        if re.search(r"\b(start over|reset|restart|new patient|clear (everything|my symptoms|chat))\b", t):
            intent = "reset"
        if re.search(r"\b(that is all|thats all|thats it|that is it|no more symptoms|nothing else|what could it be|what is wrong with me|what do i have|give me the result|diagnose)\b", t):
            intent = "diagnose"

        info_kw = bool(re.search(r"\b(what is|what are|tell me about|explain|define|how to|how can|how do|prevent\w*|precautions?|which doctor|what doctor|which specialist|specialist|medicine|medicines|tablet|tablets|drug|dose|dosage|syrup|tips|causes? of|home remedies|remedy)\b", t))
        if disease and not pos and re.search(r"\b(do i have|am i having|i think i have|i might have|maybe i have|could i have|is it|can it be|i have been diagnosed|i have)\b", t):
            st["last_disease"] = disease
            r = self._disease_info(disease)
            r["reply"] += "\n\nI cannot confirm whether *you* have it - only a doctor's tests can. Tell me your symptoms and I will compare them with this condition."
            r["quick_replies"] = ["Check my symptoms"] + r["quick_replies"][:2]
            return r
        informational = intent in ("disease_info", "precautions", "specialist", "medication", "health_tips") and (info_kw or bool(disease) or (intent == "health_tips" and not (pos or neg)))
        if (pos or neg) and not informational and intent in ("disease_info", "precautions", "specialist", "medication"):
            intent = "symptom_report"
        personal = bool(re.search(r"\b(i have|i am having|i feel|im feeling|i am feeling|my \w+( \w+)? (hurts|is paining|is hurting|pain)|i am suffering|suffering from|i got|since (yesterday|morning|last|two|three|\d))\b", t))

        # 4. symptom reporting -------------------------------------------------------------
        if (pos or neg) and (not informational or personal) and intent not in ("reset",):
            added_pos, added_neg = self._add(st, pos, neg)
            if informational:
                note = self._info_reply(intent, disease, st, t, pos)
                extra = (f"\n\nI have also noted your symptoms: **{join_list(shown(st['pos']))}**. "
                         "Say *\"what could it be\"* and I will analyse them.")
                out = dict(note); out["reply"] = note["reply"] + extra
                return out
            if not pos and neg:
                st["last_result"] = False
                return self._out(f"Noted - you do **not** have {join_list(neg)}. What symptoms *do* you have?", COMMON_START)
            return self._assess(st)

        # 5. other intents -------------------------------------------------------------------
        if intent == "reset":
            self.reset(sid)
            return self._out("Okay, I have cleared everything. Let's start fresh - how are you feeling today?", COMMON_START)
        if intent == "diagnose":
            if not st["pos"]:
                return self._out("I don't have any symptoms from you yet. Please tell me what you are feeling, e.g. *\"I have fever and body ache\"*.", COMMON_START)
            return self._assess(st, force=True)
        if disease and intent in ("symptom_report", "smalltalk", "greeting", "help") or (disease and conf < 0.5):
            if re.search(r"\b(i think|i might|do i have|maybe i have|i have|am i)\b", t):
                st["last_disease"] = disease
                r = self._disease_info(disease)
                r["reply"] += "\n\nI cannot confirm whether *you* have it. Tell me your symptoms and I will compare them."
                return r
            intent = "disease_info"
        if informational or intent in ("disease_info", "precautions", "specialist", "medication", "health_tips"):
            return self._info_reply(intent, disease, st, t, pos)
        if intent == "symptom_report":
            return self._out("I'm sorry you are not feeling well. Please tell me your symptoms - for example *\"I have fever, headache and body pain\"*.", COMMON_START)
        if conf < 0.30:
            return self._out("I'm not sure I understood that. I can **check symptoms**, explain a **disease**, give **precautions**, "
                             "suggest which **doctor** to see, or calculate your **BMI**. Try: *\"I have cough and sore throat\"*.", COMMON_START)
        return self._simple(intent, st)

    # ---------------------------------------------------------------- state updates
    def _add(self, st, pos, neg):
        new_pos = [p for p in pos if p not in st["pos"]]
        new_neg = [n for n in neg if n not in st["neg"]]
        for p in new_pos:
            st["pos"].append(p)
            if p in st["neg"]:
                st["neg"].remove(p)
        for n in new_neg:
            if n in st["pos"]:
                st["pos"].remove(n)
            st["neg"].append(n)
        return new_pos, new_neg

    @staticmethod
    def _yes_no(t):
        if re.match(r"^(not sure|unsure|maybe|dont know|i dont know|idk|cant say|not certain|no idea|do not know)\b", t):
            return "unsure"
        if re.match(r"^(yes|yeah|yep|yup|ya|yea|y|sure|definitely|correct|right|true|a little|little|slightly|somewhat|sometimes|i do|i am|i have|also|ok yes|mild|a bit)\b", t):
            return "yes"
        if re.match(r"^(no|nope|nah|n|not really|never|i dont|i do not|dont|none|neither|nothing|absent|not at all)\b", t):
            return "no"
        return None

    def _find_disease(self, t):
        m = self.alias_re.search(t)
        return self.alias_map[m.group(0)] if m else None

    # ---------------------------------------------------------------- reasoning
    def posterior(self, pos, neg):
        logs = {}
        for name, d in DISEASES.items():
            lp = math.log(PRIOR.get(name, 1.0))
            for s in pos:
                p = d["symptoms"].get(s)
                lp += math.log(p) if p else math.log(0.04)
            for s in neg:
                p = d["symptoms"].get(s)
                if p:
                    lp += math.log(1 - 0.9 * p)
            logs[name] = lp
        T = 0.75   # temperature: avoids over-confident percentages from just a few symptoms
        mx = max(logs.values())
        ex = {k: math.exp(T * (v - mx)) for k, v in logs.items()}
        z = sum(ex.values())
        return sorted(((k, v / z) for k, v in ex.items()), key=lambda kv: -kv[1])

    def _next_question(self, st, post):
        """choose the unasked symptom whose answer would split the top candidates the most"""
        known = set(st["pos"]) | set(st["neg"]) | set(st["asked"])
        top = post[:6]
        best, best_score = None, 0
        cands = {s for name, _ in top for s in DISEASES[name]["symptoms"]}
        for s in cands:
            if s in known:
                continue
            p_yes = sum(w * DISEASES[n]["symptoms"].get(s, 0.04) for n, w in top) / max(sum(w for _, w in top), 1e-9)
            score = p_yes * (1 - p_yes)
            if score > best_score:
                best, best_score = s, score
        return best

    def _assess(self, st, force=False):
        st["last_result"] = False
        if not st["pos"]:
            return self._out("What symptoms are you experiencing?", COMMON_START)
        post = self.posterior(st["pos"], st["neg"])
        top_p = post[0][1]
        enough = (top_p >= 0.60 and len(st["pos"]) >= 2) or top_p >= 0.80
        if not force and not enough and st["followups"] < MAX_FOLLOWUPS:
            q = self._next_question(st, post)
            if q:
                st["pending"] = q
                st["followups"] += 1
                lead = ""
                if "chest pain" in st["pos"] and not st.get("chest_warned"):
                    st["chest_warned"] = True
                    lead += "⚠️ Chest pain should be checked by a doctor promptly. If it is severe, spreads to your arm/jaw, or comes with sweating or breathlessness, call 112 now.\n\n"
                if st["followups"] == 1:
                    lead += f"Thanks, I have noted: **{join_list(shown(st['pos']))}**.\nLet me ask a few quick questions to narrow it down.\n\n"
                return self._out(f"{lead}Are you also experiencing **{q}**?", ["Yes", "No", "Not sure", "That's all - show result"])
        return self._result(st, post)

    # ---------------------------------------------------------------- response builders
    def _out(self, reply, quick=None, **extra):
        d = {"reply": reply, "quick_replies": quick or []}
        d.update(extra)
        return d

    def _result(self, st, post):
        st["last_result"] = True
        top = [(n, p) for n, p in post if p >= 0.04][:3] or post[:1]
        name, prob = top[0]
        d = DISEASES[name]
        st["last_disease"] = name
        lines = ["### Symptom analysis",
                 f"**Symptoms reported:** {join_list(shown(st['pos']))}"]
        if st["neg"]:
            lines.append(f"**Not present:** {join_list(st['neg'])}")
        lines += ["", "**Possible conditions** (likelihood among the conditions I know):"]
        for i, (n, p) in enumerate(top, 1):
            lines.append(f"{i}. **{n}** - {round(p * 100)}%")
        if prob < 0.40:
            lines.append("\n_Your symptoms are quite general, so I cannot narrow it down further. Adding more symptoms helps._")
        lines += ["", f"**About {name}:** {d['about']}", "", "**What you can do now:**"]
        lines += [f"• {x}" for x in d["self_care"]]
        urg = {"mild": "This is usually manageable at home. See a doctor if it lasts more than 3-5 days or gets worse.",
               "moderate": "Please consult a doctor within a day or two for proper evaluation.",
               "high": "Please see a doctor **as soon as possible (within 24 hours)** - this needs proper tests and treatment."}[d["urgency"]]
        lines += ["", f"**See a doctor:** {d['specialist']}. {urg}",
                  f"**Go to the emergency room immediately if you have:** {d['red_flags']}."]
        if "chest pain" in st["pos"]:
            lines += ["", "⚠️ Chest pain should always be checked by a doctor promptly. Seek emergency care if it is severe, spreads to the arm or jaw, or comes with sweating or breathlessness."]
        if "high fever" in st["pos"] and d["urgency"] != "high":
            lines += ["", "⚠️ A very high fever (above 102 F / 39 C) lasting more than a day should be examined by a doctor."]
        lines += ["", DISCLAIMER]
        short = d["short"]
        return self._out("\n".join(lines),
                         [f"How to prevent {short}", "Which doctor should I see?", "Add more symptoms", "Start over"],
                         conditions=[{"name": n, "probability": round(p, 3)} for n, p in top],
                         symptoms=shown(st["pos"]))

    def _disease_info(self, name):
        d = DISEASES[name]
        reply = (f"### {name}\n{d['about']}\n\n"
                 f"**Common symptoms:** {join_list(shown(sorted(d['symptoms'], key=lambda s: -d['symptoms'][s]))[:7])}\n\n"
                 f"**Prevention / precautions:**\n" + "\n".join(f"• {p}" for p in d["precautions"]) +
                 f"\n\n**Which doctor:** {d['specialist']}\n\n{DISCLAIMER}")
        return self._out(reply, [f"Which doctor for {d['short']}?", f"Self-care for {d['short']}", "Check my symptoms"])

    def _info_reply(self, intent, disease, st, t, pos=None):
        pos = pos or []
        name = disease or st.get("last_disease")
        if intent == "disease_info":
            if not name:
                return self._out("Which condition would you like to know about? For example: *\"What is dengue?\"* or *\"Tell me about diabetes\"*.",
                                 ["What is diabetes?", "What is malaria?", "What is asthma?"])
            st["last_disease"] = name
            return self._disease_info(name)
        if intent == "precautions":
            if not name:
                return self._out("For which condition do you want precautions? e.g. *\"How to prevent dengue?\"*. Or tell me your symptoms.", ["How to prevent dengue?", "How to prevent flu?", "Check my symptoms"])
            st["last_disease"] = name
            d = DISEASES[name]
            return self._out(f"### Precautions & care for {name}\n**Prevention:**\n" + "\n".join(f"• {p}" for p in d["precautions"]) +
                             "\n\n**Self-care when affected:**\n" + "\n".join(f"• {p}" for p in d["self_care"]) +
                             f"\n\n**Watch out - see a doctor urgently for:** {d['red_flags']}.\n\n{DISCLAIMER}",
                             [f"Which doctor for {d['short']}?", "Check my symptoms"])
        if intent == "specialist":
            if name:
                st["last_disease"] = name
                d = DISEASES[name]
                return self._out(f"For **{name}**, you should consult a **{d['specialist']}**.\n\nSee a doctor urgently if you have: {d['red_flags']}.", [f"How to prevent {d['short']}", "Check my symptoms"])
            if pos:
                docs = []
                for sym in pos:
                    sp = SYMPTOM_SPECIALIST.get(sym)
                    if sp and sp not in docs:
                        docs.append(sp)
                if docs:
                    return self._out(f"For **{join_list(shown(pos))}**, a good first step is a **{docs[0]}**" +
                                     (f" (you may also be referred to: {join_list(docs[1:])})" if docs[1:] else "") +
                                     ".\nWhen unsure, start with a **General Physician**.\n\nWant me to check what it could be? Tell me your other symptoms.",
                                     ["Check my symptoms", "Health tips"])
            return self._out("Tell me your symptoms and I will suggest the right specialist - for example *\"I have a skin rash and itching\"*. "
                             "As a general rule, start with a **General Physician**, who will refer you to a specialist if needed.", COMMON_START)
        if intent == "medication":
            return self._out(
                "I can't prescribe medicines or give doses, because the right medicine depends on your age, weight, allergies, pregnancy and other conditions.\n\n"
                "**General safe guidance:**\n"
                "• For simple fever or pain, many people use paracetamol (acetaminophen) exactly as written on the label or as advised by a pharmacist/doctor.\n"
                "• Do **not** take antibiotics without a prescription - they do not work for viral colds, cough or flu.\n"
                "• Avoid aspirin for children and teenagers, and avoid ibuprofen/aspirin if dengue is possible.\n"
                "• Tell your doctor about any allergies, pregnancy, kidney, liver or stomach problems.\n\n"
                "Tell me your symptoms and I will suggest which doctor to see.\n\n" + DISCLAIMER, ["Check my symptoms", "Which doctor should I see?"])
        if intent == "health_tips":
            return self._out(
                "### Everyday health tips\n"
                "• **Sleep:** 7-9 hours at a fixed time; keep screens away 1 hour before bed.\n"
                "• **Water:** about 2-3 litres a day (more in hot weather or when sick).\n"
                "• **Food:** plenty of vegetables, fruit, whole grains and protein; limit sugar, salt, fried and packaged food.\n"
                "• **Exercise:** at least 30 minutes of brisk activity on most days; stretch and move every hour if you sit a lot.\n"
                "• **Stress:** deep breathing, short walks, hobbies, and talking to someone you trust.\n"
                "• **Immunity:** stay up to date with vaccines, wash hands, avoid smoking and limit alcohol.\n"
                "• **Check-ups:** blood pressure, blood sugar and weight at least once a year.\n\n" + DISCLAIMER,
                ["Calculate my BMI", "Check my symptoms"])
        return self._out("How can I help you?", COMMON_START)

    def _simple(self, intent, st):
        if intent == "greeting":
            return self._out("Hello! I'm **MediBot**, your AI health assistant. 👋\nTell me how you are feeling - for example *\"I have fever and headache\"* - and I will help you understand what it could be.",
                             COMMON_START + ["Calculate BMI"])
        if intent == "goodbye":
            return self._out("Take care and get well soon! 🌿 Remember to see a doctor if your symptoms get worse.")
        if intent == "thanks":
            return self._out("You're welcome! Is there anything else I can help you with?", ["Check my symptoms", "Health tips", "Calculate BMI"])
        if intent == "help":
            return self._out("I'm **MediBot**. I can:\n• **Check symptoms** - tell me what you feel, I'll ask a few questions and suggest possible conditions\n"
                             "• **Explain diseases** - e.g. *\"What is dengue?\"*\n• Give **precautions & self-care** - e.g. *\"How to prevent malaria?\"*\n"
                             "• Suggest **which doctor** to consult\n• Calculate your **BMI**\n• Share everyday **health tips**\n\n"
                             "I always flag **emergencies**. I give information only, not a diagnosis.",
                             COMMON_START + ["Calculate BMI"])
        if intent == "smalltalk":
            return self._out("I'm an AI assistant, not a doctor, so I can't diagnose you - but I can compare your symptoms with common conditions, "
                             "explain them and tell you when to see a doctor. How are you feeling today?", COMMON_START)
        return self._out("How can I help you today?", COMMON_START)

    def _emergency(self, kind):
        if kind == "suicide":
            msg = ("I'm really sorry you are going through this - you are not alone, and your life matters. 💙\n\n"
                   "**Please reach out for support right now:**\n"
                   "• **Tele-MANAS (India, free, 24x7):** 14416 or 1-800-891-4416\n"
                   "• **Emergency:** 112\n"
                   "• If you are outside India, please call your local emergency number or a crisis line.\n\n"
                   "If you are in immediate danger, call emergency services or go to the nearest hospital. "
                   "Please tell a family member or friend who can stay with you. I'm here to listen too - would you like to tell me what is happening?")
            return self._out(msg, ["I'm safe now", "Start over"], emergency=True, emergency_type=kind)
        what = {"heart": "possible heart attack", "breathing": "severe breathing difficulty", "stroke": "possible stroke",
                "unconscious": "loss of consciousness / seizure", "bleeding": "serious bleeding", "allergy": "severe allergic reaction",
                "poison": "possible poisoning / overdose"}[kind]
        extra = {"stroke": "Note the time symptoms started. Do **not** give food or drink. (Remember FAST: Face drooping, Arm weakness, Speech difficulty, Time to call.)",
                 "heart": "Sit down and stay calm. If you are not allergic and a doctor has not told you otherwise, chewing one regular aspirin can help - but only while waiting for emergency help.",
                 "breathing": "Sit upright, loosen tight clothing, and stay as calm as you can.",
                 "bleeding": "Press firmly on the wound with a clean cloth and do not remove it.",
                 "allergy": "If you have an adrenaline auto-injector, use it now.",
                 "unconscious": "Lay the person on their side, do not put anything in their mouth, and keep them safe from injury.",
                 "poison": "Do not make the person vomit. Keep the container/pack to show the doctor. In India, poison helpline: 1066 (AIIMS)."}[kind]
        msg = (f"🚨 **This could be an emergency ({what}).**\n\n"
               f"**Call 112 or 108 (ambulance) or go to the nearest hospital emergency room immediately.** Do not wait or travel alone.\n\n"
               f"{extra}\n\n{HELPLINES}\n\n_I cannot assess emergencies - please get urgent medical help._")
        return self._out(msg, ["I'm safe now", "Start over"], emergency=True, emergency_type=kind)

    # ---------------------------------------------------------------- BMI
    @staticmethod
    def _parse_bmi(raw):
        t = raw.lower()
        w = None
        m = re.search(r"(\d{2,3}(?:\.\d+)?)\s*(?:kg|kgs|kilograms?|kilos?)\b", t)
        if m:
            w = float(m.group(1))
        else:
            m = re.search(r"(\d{2,3}(?:\.\d+)?)\s*(?:lb|lbs|pounds?)\b", t)
            if m:
                w = float(m.group(1)) * 0.45359237
        h = None
        m = re.search(r"(\d{2,3}(?:\.\d+)?)\s*(?:cm|centimet\w+)\b", t)
        if m:
            h = float(m.group(1)) / 100
        if h is None:
            m = re.search(r"\b(\d)\s*(?:ft|feet|foot|')\s*(\d{1,2})?\s*(?:in|inch|inches|\")?", t)
            if m:
                h = (int(m.group(1)) * 12 + int(m.group(2) or 0)) * 0.0254
        if h is None:
            m = re.search(r"\b(\d(?:\.\d+)?)\s*(?:m|meters?|metres?)\b", t)
            if m:
                h = float(m.group(1))
        if w and h and 1.0 <= h <= 2.5 and 15 <= w <= 300:
            return w, h
        return None

    def _bmi_prompt(self, st):
        st["awaiting_bmi"] = True
        return self._out("Sure! Tell me your **weight and height**, for example: *\"70 kg, 170 cm\"* (you can also use feet/inches or pounds).",
                         ["65 kg 165 cm", "80 kg 175 cm"])

    def _bmi_reply(self, w, h):
        bmi = w / (h * h)
        if bmi < 18.5: cat, advice = "Underweight", "Consider a balanced, calorie-adequate diet with protein, and check with a doctor/dietitian if unintentional."
        elif bmi < 25: cat, advice = "Normal weight", "Great! Maintain it with a balanced diet and regular exercise."
        elif bmi < 30: cat, advice = "Overweight", "Aim for gradual weight loss with a healthier diet and 30 minutes of activity daily."
        else: cat, advice = "Obese", "Please consult a doctor/dietitian for a safe weight-loss plan; even a 5-10% loss improves health."
        lo, hi = 18.5 * h * h, 24.9 * h * h
        return self._out(f"### Your BMI: **{bmi:.1f}** - {cat}\nWeight {w:.1f} kg, height {h * 100:.0f} cm.\n\n{advice}\n\n"
                         f"Healthy weight range for your height: **{lo:.0f}-{hi:.0f} kg**.\n"
                         "_Note: for South Asian adults, health risks begin at a lower BMI (about 23 and above). BMI does not measure body fat directly._",
                         ["Health tips", "Check my symptoms"], bmi=round(bmi, 1), category=cat)
