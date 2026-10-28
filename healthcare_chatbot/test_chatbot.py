"""
Automated tests - run with:  python test_chatbot.py      (or: pytest -q)
Each case sends real messages through the engine and checks the answer.
"""
import sys
from chatbot import HealthBot

bot = HealthBot()
_n = 0


def ask(*msgs):
    global _n
    _n += 1
    sid = f"t{_n}"
    r = None
    for m in msgs:
        r = bot.respond(sid, m)
    return r


def top(r):
    return r["conditions"][0]["name"] if r.get("conditions") else None


# ---------------- symptom -> condition -------------------------------------------------
def test_malaria():          assert top(ask("I have high fever with chills and sweating")) == "Malaria"
def test_dengue():           assert top(ask("high fever, joint pain, muscle pain and pain behind my eyes")) == "Dengue"
def test_flu_flow():         assert top(ask("my head hurts a lot", "yes", "no", "yes", "yes")).startswith("Influenza")
def test_diabetes():         assert top(ask("always thirsty, urinate a lot, blurry vision and losing weight")).startswith("Diabetes")
def test_uti():              assert top(ask("burning while peeing and lower abdominal pain")).startswith("Urinary")
def test_jaundice():         assert top(ask("yellow eyes, dark urine and no appetite")).startswith("Jaundice")
def test_allergy():          assert top(ask("sneezing, runny nose and watery itchy eyes")).startswith("Allergic")
def test_gerd():             assert top(ask("heartburn and sour taste after meals")).startswith("Acid")
def test_asthma():           assert top(ask("wheezing and shortness of breath")) == "Asthma"
def test_anemia():           assert top(ask("i feel tired and look pale and dizzy")).startswith("Anemia")
def test_gastro():           assert top(ask("vomiting, diarrhea and stomach pain after eating outside food")).startswith("Gastro")
def test_chickenpox():       assert top(ask("itchy blisters and red spots with mild fever")) == "Chickenpox"
def test_tb():               assert top(ask("cough for more than 3 weeks with night sweats and weight loss")).startswith("Tuberculosis")
def test_migraine():         assert top(ask("headache, sensitive to light and nausea")) == "Migraine"
def test_pinkeye():          assert top(ask("red eyes with sticky discharge")).startswith("Conjunctivitis")
def test_typos():
    pos, _ = bot.extractor.extract("i have feever, headach and vomitting")
    assert {"fever", "headache", "vomiting"} <= set(pos)

# ---------------- understanding -------------------------------------------------------
def test_negation():
    r = ask("I have fever and cough but no sore throat", "that's all")
    assert "sore throat" in r["reply"] and "Not present" in r["reply"]

def test_follow_up_question():
    r = ask("I have fever")
    assert "Are you also experiencing" in r["reply"] and "Yes" in r["quick_replies"]

def test_no_symptoms_yet():
    assert "symptoms" in ask("what could it be")["reply"].lower()

# ---------------- safety ----------------------------------------------------------------
def test_emergency_heart():  assert ask("I have severe chest pain and sweating").get("emergency")
def test_emergency_breath(): assert ask("my father can't breathe")["emergency_type"] == "breathing"
def test_emergency_stroke(): assert ask("face drooping and slurred speech")["emergency_type"] == "stroke"
def test_selfharm():
    r = ask("i want to end my life")
    assert r.get("emergency") and "14416" in r["reply"]
def test_mild_chest_pain_warns():
    assert "Chest pain" in ask("i have chest pain since morning")["reply"]

# ---------------- other intents -------------------------------------------------------
def test_greeting():         assert "MediBot" in ask("hello")["reply"]
def test_goodbye():          assert "Take care" in ask("bye")["reply"]
def test_disease_info():     assert "Dengue" in ask("what is dengue?")["reply"]
def test_precautions():      assert "mosquito" in ask("how to prevent dengue")["reply"].lower()
def test_followup_it():      assert "Precautions" in ask("what is malaria", "how to prevent it")["reply"]
def test_specialist():       assert "Dermatologist" in ask("which doctor for skin rash and itching")["reply"]
def test_medicine_safety():
    r = ask("what medicine should i take for fever")["reply"]
    assert "can't prescribe" in r and "antibiotics" in r.lower()
def test_bmi():
    r = ask("calculate my bmi", "70 kg 170 cm")
    assert r["bmi"] == 24.2
def test_bmi_feet():         assert ask("i weigh 82 kg and my height is 5 ft 9 in")["category"] == "Overweight"
def test_health_tips():      assert "Sleep" in ask("give me health tips")["reply"]
def test_gibberish():        assert "not sure I understood" in ask("asdfgh qwerty")["reply"]
def test_reset():
    r = ask("I have fever", "start over")
    assert "fresh" in r["reply"] or "cleared" in r["reply"]
def test_do_i_have():        assert "cannot confirm" in ask("do i have dengue")["reply"]


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in fns:
        try:
            fn(); print(f"  PASS  {name}")
        except Exception as e:
            failed += 1; print(f"  FAIL  {name}: {type(e).__name__} {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} tests passed")
    sys.exit(1 if failed else 0)
