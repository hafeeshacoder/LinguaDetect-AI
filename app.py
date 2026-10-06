import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# ---------------------------------------------------------
# PAGE SETTINGS
# ---------------------------------------------------------

st.set_page_config(
    page_title="LinguaDetect AI",
    page_icon="🌐",
    layout="wide"
)

# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

st.html("""
<style>

body {
    background-color: #f5f7ff;
}

.main-title {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    padding: 40px;
    border-radius: 25px;
    text-align: center;
    color: white;
    margin-bottom: 30px;
}

.main-title h1 {
    font-size: 42px;
    margin: 0;
}

.main-title p {
    font-size: 18px;
    margin-top: 10px;
}

.info-card {
    background: white;
    padding: 25px;
    border-radius: 18px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 5px 20px rgba(0,0,0,0.05);
    min-height: 180px;
}

.info-card h3 {
    color: #111827;
}

.info-card p {
    color: #4b5563;
    line-height: 1.7;
}

.language-card {
    background: white;
    padding: 20px;
    border-radius: 16px;
    text-align: center;
    border: 1px solid #e5e7eb;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
}

.language-card .flag {
    font-size: 35px;
}

.language-card h4 {
    margin: 8px 0;
}

.result-card {
    background: #ecfdf5;
    border: 1px solid #bbf7d0;
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    margin-top: 25px;
}

.result-card h2 {
    color: #166534;
    font-size: 35px;
}

.footer {
    text-align: center;
    color: #6b7280;
    padding: 30px;
}

</style>
""")

# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.html("""
<div class="main-title">
    <h1>🌐 LinguaDetect AI</h1>
    <p>Intelligent Language Detection using Natural Language Processing</p>
</div>
""")

# ---------------------------------------------------------
# TRAINING DATA
# ---------------------------------------------------------

# English
english_texts = [
    "hello how are you",
    "good morning",
    "good evening",
    "what is your name",
    "how are you today",
    "i am learning artificial intelligence",
    "i love programming",
    "this is a beautiful day",
    "where are you going",
    "have a nice day"
]

# Tamil
tamil_texts = [
    "வணக்கம் எப்படி இருக்கிறீர்கள்",
    "நான் நன்றாக இருக்கிறேன்",
    "தமிழ் மொழி மிகவும் அழகானது",
    "உங்கள் பெயர் என்ன",
    "நீங்கள் எப்படி இருக்கிறீர்கள்",
    "நான் செயற்கை நுண்ணறிவு கற்கிறேன்",
    "எனக்கு தமிழ் பிடிக்கும்",
    "இன்று நல்ல நாள்",
    "நீங்கள் எங்கு செல்கிறீர்கள்",
    "நன்றி"
]

# French
french_texts = [
    "bonjour comment allez vous",
    "bonsoir",
    "comment allez vous aujourd'hui",
    "quel est votre nom",
    "je suis heureux",
    "j'aime apprendre",
    "j'aime programmer",
    "bonne journée",
    "merci beaucoup",
    "au revoir"
]

# Spanish
spanish_texts = [
    "Contigo, siempre",
    "Te amo",
    "Mi amor",
    "Mi vida",
    "Mi corazón",
    "Para siempre",
    "Tú y yo",
    "Solo nosotros",
    "Siempre contigo",
    "Eres mi felicidad",
    "Eres mi mundo",
    "Amor de mi vida",
    "Almas gemelas",
    "Juntos para siempre",
    "Hasta el final",
    "hola como estas",
    "buenos dias",
    "buenas tardes",
    "cual es tu nombre",
    "como estas hoy",
    "me gusta aprender",
    "me gusta programar",
    "que tengas un buen dia",
    "muchas gracias",
    "hasta luego"
]

# German
german_texts = [
    "Für immer",
    "Für immer wir",
    "Nur wir zwei",
    "Du und ich",
    "Mit dir",
    "Immer bei dir",
    "Bis ans Ende",
    "Mein Zuhause",
    "Mein Herz",
    "Meine Welt",
    "Ewige Liebe",
    "Du bist mein Glück",
    "Liebe meines Lebens",
    "Für immer an deiner Seite",
    "Gemeinsam für immer",
    "Nur wir zwei",
    "hallo wie geht es dir",
    "guten morgen",
    "guten abend",
    "wie heißt du",
    "wie geht es dir heute",
    "ich lerne künstliche intelligenz",
    "ich liebe programmieren",
    "einen schönen tag",
    "vielen dank",
    "auf wiedersehen"
]

# ---------------------------------------------------------
# COMBINE TRAINING DATA
# ---------------------------------------------------------

texts = (
    english_texts
    + tamil_texts
    + french_texts
    + spanish_texts
    + german_texts
)

languages = (
    ["English"] * len(english_texts)
    + ["Tamil"] * len(tamil_texts)
    + ["French"] * len(french_texts)
    + ["Spanish"] * len(spanish_texts)
    + ["German"] * len(german_texts)
)

# Safety check
assert len(texts) == len(languages)

# ---------------------------------------------------------
# NLP MODEL
# ---------------------------------------------------------

vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 5)
)

X = vectorizer.fit_transform(texts)

model = MultinomialNB()

model.fit(X, languages)

# ---------------------------------------------------------
# INFORMATION CARDS
# ---------------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    st.html("""
    <div class="info-card">

        <h3>🧠 How does it work?</h3>

        <p>
        Enter a sentence and the NLP model analyzes the
        text and predicts its language.
        </p>

        <p>
        <b>TF-IDF Character N-Grams</b> are used for
        feature extraction and
        <b>Multinomial Naive Bayes</b> is used for
        classification.
        </p>

    </div>
    """)

with col2:

    st.html("""
    <div class="info-card">

        <h3>⚡ NLP Pipeline</h3>

        <p>📝 Input Text</p>
        <p>↓</p>
        <p>🔢 TF-IDF Features</p>
        <p>↓</p>
        <p>🤖 Naive Bayes</p>
        <p>↓</p>
        <p>🌐 Language Prediction</p>

    </div>
    """)

# ---------------------------------------------------------
# INPUT
# ---------------------------------------------------------

st.subheader("✍️ Enter Your Text")

user_text = st.text_area(
    "Enter text",
    height=150,
    placeholder="Example: Hello, how are you today?",
    label_visibility="collapsed"
)

# ---------------------------------------------------------
# BUTTONS
# ---------------------------------------------------------

col1, col2, col3 = st.columns([1, 1, 3])

with col1:

    detect = st.button(
        "🔍 Detect Language",
        use_container_width=True
    )

with col2:

    clear = st.button(
        "🗑️ Clear",
        use_container_width=True
    )

# ---------------------------------------------------------
# CLEAR BUTTON
# ---------------------------------------------------------

if clear:
    st.session_state["clear_text"] = True
    st.rerun()

# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

if detect:

    if not user_text.strip():

        st.warning("⚠️ Please enter some text.")

    else:

        # Convert input text into TF-IDF features
        user_vector = vectorizer.transform([user_text])

        # Predict language
        prediction = model.predict(user_vector)[0]

        # Get probabilities
        probabilities = model.predict_proba(user_vector)[0]

        # Highest probability
        confidence = max(probabilities) * 100

        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        st.html(f"""
        <div class="result-card">

            <p>DETECTED LANGUAGE</p>

            <h2>🌐 {prediction}</h2>

            <p>
                Model Confidence:
                <b>{confidence:.2f}%</b>
            </p>

        </div>
        """)

        st.progress(
            min(int(confidence), 100)
        )

        # -------------------------------------------------
        # PREDICTION DETAILS
        # -------------------------------------------------

        st.subheader("📊 Prediction Details")

        results = []

        for language, probability in zip(
            model.classes_,
            probabilities
        ):

            results.append(
                (language, probability * 100)
            )

        # Sort highest to lowest
        results.sort(
            key=lambda x: x[1],
            reverse=True
        )

        for language, probability in results:

            st.write(
                f"**{language}** — {probability:.2f}%"
            )

            st.progress(
                min(int(probability), 100)
            )

# ---------------------------------------------------------
# SUPPORTED LANGUAGES
# ---------------------------------------------------------

st.subheader("🌍 Supported Languages")

col1, col2, col3, col4, col5 = st.columns(5)

language_data = [
    ("🇬🇧", "English"),
    ("🇮🇳", "Tamil"),
    ("🇫🇷", "French"),
    ("🇪🇸", "Spanish"),
    ("🇩🇪", "German")
]

columns = [
    col1,
    col2,
    col3,
    col4,
    col5
]

for column, (flag, language) in zip(
    columns,
    language_data
):

    with column:

        st.html(f"""
        <div class="language-card">

            <div class="flag">{flag}</div>

            <h4>{language}</h4>

            <p>Supported</p>

        </div>
        """)

# ---------------------------------------------------------
# EXAMPLES
# ---------------------------------------------------------

st.subheader("💡 Try These Examples")

examples = [
    ("🇬🇧 English", "Hello, how are you today?"),
    ("🇮🇳 Tamil", "வணக்கம் எப்படி இருக்கிறீர்கள்?"),
    ("🇫🇷 French", "Bonjour comment allez vous?"),
    ("🇪🇸 Spanish", "Hola como estas?"),
    ("🇩🇪 German", "Hallo wie geht es dir?")
]

for language, example in examples:

    st.info(
        f"**{language}** → {example}"
    )

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.html("""
<div class="footer">

    🌐 LinguaDetect AI

    <br><br>

    Built with Python • Streamlit • NLP

    <br><br>

    TF-IDF Character N-Grams + Multinomial Naive Bayes

</div>
""")
