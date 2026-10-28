# AI-Powered Healthcare Chatbot (MediBot)

A conversational health assistant that **understands what you type, asks follow-up questions, and responds
according to your input**: it checks symptoms, suggests possible conditions with likelihoods, explains diseases,
gives precautions and the right specialist, calculates BMI, and detects medical emergencies.

> Educational project. It gives general information only - it is **not** a medical diagnosis and never replaces a doctor.

## Run it
```bash
pip install -r requirements.txt
python app.py            # web chat  ->  open http://127.0.0.1:5000
python cli.py            # or chat in the terminal
python test_chatbot.py   # 37 automated tests
```

## What you can type (examples)
| You type | MediBot does |
|---|---|
| `I have high fever, chills and sweating` | analyses symptoms -> **Malaria** (+ likelihood bars, self-care, doctor, red flags) |
| `my head hurts a lot` | asks smart yes/no follow-ups (fever? cough? body ache?) then gives the result |
| `no fever but bad cough and sore throat` | understands **negation** (fever excluded) |
| `i have feever and vomitting` | tolerates **typos** |
| `what is dengue?` / `how to prevent it` / `which doctor for it` | disease info, precautions, specialist (remembers the topic) |
| `calculate my bmi` -> `70 kg 170 cm` | BMI, category, healthy weight range |
| `what medicine for fever` | safe guidance only - no doses, no prescriptions |
| `severe chest pain and sweating` | **emergency alert**: call 112 / 108 |
| `i want to end my life` | supportive reply with Tele-MANAS 14416 and 112 |
| `health tips`, `are you a doctor?`, `bye` | tips, honest self-description, goodbye |

## How it works (pipeline)
1. **Safety layer** - regex red-flag rules (heart attack, stroke, breathing trouble, bleeding, poisoning, self-harm ...) run first.
2. **Intent classification (ML)** - TF-IDF (word + character n-grams) + Logistic Regression, trained at start-up on 230+ example phrases, 14 intents.
3. **Entity extraction** - 59 symptoms with ~700 everyday phrasings, typo correction (`difflib`), negation handling, disease names, BMI values.
4. **Dialogue manager** - per-session memory; picks the next question that best separates the top candidate diseases (information-gain style).
5. **Reasoning** - naive-Bayes-style scoring over 21 conditions: `log prior + sum log P(symptom | disease)`, with penalties for explicitly absent symptoms, and a softmax temperature so a few symptoms never give a falsely certain answer.
6. **Response builder** - about, self-care, precautions, specialist, urgency level, ER red flags, disclaimer. The web UI renders likelihood bars, quick-reply chips, emergency styling, a mobile layout and dark mode.

## Project structure
```
app.py                Flask server  (/ , /api/chat , /api/reset , /api/health)
chatbot.py            NLU + dialogue + reasoning engine
knowledge_base.py     symptoms, 21 diseases, intents, emergency rules  (edit this to extend the bot)
cli.py                terminal client
templates/index.html  chat UI
test_chatbot.py       37 tests
```

## Conditions covered
Common cold, Influenza, COVID-19, Malaria, Dengue, Typhoid, Gastroenteritis / food poisoning, Migraine, Allergic rhinitis,
Asthma, Hypertension, Type 2 diabetes, Pneumonia, UTI, Jaundice / Hepatitis, Anemia, Chickenpox, Tuberculosis, Acid reflux (GERD),
Sinusitis, Conjunctivitis.

## Limitations (state these honestly in your report / viva)
* The symptom probabilities in `knowledge_base.py` are hand-written approximations from general medical knowledge, **not** learned from patient data, and have not been clinically validated.
* Only 21 common conditions are covered; anything else is mapped to the closest one - hence the disclaimers and "see a doctor" advice.
* Age, gender, pregnancy, symptom duration and medical history are not modelled.
* Percentages are relative likelihoods among the known conditions, not real-world probabilities.

## Ideas to extend
Train the symptom model on a public dataset (e.g. Kaggle "Disease Symptom Prediction"), replace the intent classifier with a fine-tuned BERT,
add Hindi / multilingual support, voice input, appointment booking, a database for chat history, and deploy on Render or Hugging Face Spaces.
