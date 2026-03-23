"""
Abstention dataset: 200 items (120 answerable + 80 unanswerable).

Answerable questions have verifiably correct answers across math, science,
geography, and history. Unanswerable questions span 5 categories (16 each).
"""

import random
import pandas as pd


# ---------------------------------------------------------------------------
# Answerable questions — Easy (40), Medium (50), Hard (30)
# ---------------------------------------------------------------------------

_EASY_QUESTIONS = [
    # Math (15)
    ("What is 7 × 8?", "56"),
    ("What is 12 + 19?", "31"),
    ("What is 100 - 37?", "63"),
    ("What is 9 × 6?", "54"),
    ("What is 144 / 12?", "12"),
    ("What is 15 + 28?", "43"),
    ("What is 50 - 23?", "27"),
    ("What is 8 × 7?", "56"),
    ("What is 81 / 9?", "9"),
    ("What is 25 + 36?", "61"),
    ("What is 200 - 45?", "155"),
    ("What is 11 × 11?", "121"),
    ("What is 72 / 8?", "9"),
    ("What is 33 + 44?", "77"),
    ("What is 6 × 9?", "54"),
    # Science (10)
    ("What is the chemical symbol for gold?", "Au"),
    ("What is the chemical symbol for oxygen?", "O"),
    ("How many planets are in our solar system?", "8"),
    ("What is the boiling point of water in degrees Celsius?", "100"),
    ("What is the chemical formula for water?", "H2O"),
    ("What gas do plants absorb from the atmosphere?", "Carbon dioxide"),
    ("What is the chemical symbol for iron?", "Fe"),
    ("What is the chemical symbol for sodium?", "Na"),
    ("How many bones are in the adult human body?", "206"),
    ("What is the speed of light in km/s, approximately?", "300000"),
    # Geography (10)
    ("What is the capital of France?", "Paris"),
    ("What is the capital of Japan?", "Tokyo"),
    ("What is the largest continent by area?", "Asia"),
    ("What is the longest river in the world?", "Nile"),
    ("What is the capital of Australia?", "Canberra"),
    ("What ocean is the largest by area?", "Pacific"),
    ("What is the capital of Canada?", "Ottawa"),
    ("What is the smallest country in the world by area?", "Vatican City"),
    ("On which continent is Brazil located?", "South America"),
    ("What is the capital of Germany?", "Berlin"),
    # History (5)
    ("In what year did World War II end?", "1945"),
    ("In what year did the United States declare independence?", "1776"),
    ("Who was the first President of the United States?", "George Washington"),
    ("In what year did the Berlin Wall fall?", "1989"),
    ("In what year did World War I begin?", "1914"),
]

_MEDIUM_QUESTIONS = [
    # Math (15)
    ("What is 347 + 589?", "936"),
    ("What is 23 × 17?", "391"),
    ("What is the square root of 196?", "14"),
    ("What is 15% of 240?", "36"),
    ("What is 1024 / 32?", "32"),
    ("What is 37 × 43?", "1591"),
    ("What is 2 to the power of 10?", "1024"),
    ("What is 999 - 573?", "426"),
    ("What is the least common multiple of 12 and 18?", "36"),
    ("What is the greatest common divisor of 48 and 36?", "12"),
    ("What is 125 × 8?", "1000"),
    ("What is the square root of 289?", "17"),
    ("What is 45% of 200?", "90"),
    ("What is 7 to the power of 3?", "343"),
    ("What is 3.5 × 2.4?", "8.4"),
    # Science (15)
    ("What is the atomic number of carbon?", "6"),
    ("What planet is known as the Red Planet?", "Mars"),
    ("What is the chemical symbol for potassium?", "K"),
    ("What is the hardest natural substance on Earth?", "Diamond"),
    ("What is the most abundant gas in Earth's atmosphere?", "Nitrogen"),
    ("What is the pH of pure water at 25°C?", "7"),
    ("What is the powerhouse of the cell?", "Mitochondria"),
    ("How many chromosomes do humans have?", "46"),
    ("What element has the atomic number 79?", "Gold"),
    ("What is the chemical formula for table salt?", "NaCl"),
    ("What is the SI unit of electric current?", "Ampere"),
    ("What type of bond involves the sharing of electrons?", "Covalent"),
    ("What is the freezing point of water in Fahrenheit?", "32"),
    ("What vitamin is produced when skin is exposed to sunlight?", "Vitamin D"),
    ("How many valence electrons does carbon have?", "4"),
    # Geography (10)
    ("What is the capital of South Korea?", "Seoul"),
    ("What is the deepest ocean trench in the world?", "Mariana Trench"),
    ("Through how many countries does the Danube River flow?", "10"),
    ("What is the highest mountain in Africa?", "Mount Kilimanjaro"),
    ("What is the capital of Turkey?", "Ankara"),
    ("What sea lies between Europe and Africa?", "Mediterranean Sea"),
    ("What is the largest desert in the world?", "Sahara"),
    ("What is the capital of New Zealand?", "Wellington"),
    ("What is the most populous country in South America?", "Brazil"),
    ("What strait separates Europe from Asia at Istanbul?", "Bosphorus"),
    # History (10)
    ("In what year was the Treaty of Westphalia signed?", "1648"),
    ("Who wrote 'The Republic'?", "Plato"),
    ("In what year did the French Revolution begin?", "1789"),
    ("What empire was ruled by Suleiman the Magnificent?", "Ottoman Empire"),
    ("In what year did the Titanic sink?", "1912"),
    ("Who invented the printing press?", "Johannes Gutenberg"),
    ("In what year was the Magna Carta signed?", "1215"),
    ("What was the last dynasty to rule China?", "Qing"),
    ("In what year did the Russian Revolution occur?", "1917"),
    ("Who was the first Emperor of Rome?", "Augustus"),
]

_HARD_QUESTIONS = [
    # Math (8)
    ("What is the largest prime factor of 2310?", "11"),
    ("What is 17 × 23 mod 13?", "1"),
    ("What is the sum of the first 20 positive integers?", "210"),
    ("How many prime numbers are there between 1 and 50?", "15"),
    ("What is the value of 12 factorial divided by 10 factorial?", "132"),
    ("What is the cube root of 2744?", "14"),
    ("What is 2^16?", "65536"),
    ("What is the sum of interior angles of a hexagon in degrees?", "720"),
    # Science (8)
    ("What is the atomic number of Rutherfordium?", "104"),
    ("What is the Avogadro constant to 3 significant figures?", "6.02e23"),
    ("What is the half-life of Carbon-14 in years, approximately?", "5730"),
    ("What is the charge of a muon in units of elementary charge?", "-1"),
    ("What element has the highest electronegativity?", "Fluorine"),
    ("What is the second most abundant element in Earth's crust?", "Silicon"),
    ("What is the escape velocity of Earth in km/s, approximately?", "11.2"),
    ("What is the name of the longest bone in the human body?", "Femur"),
    # Geography (7)
    ("What is the capital of Myanmar?", "Naypyidaw"),
    ("What is the smallest country in mainland Africa by area?", "Gambia"),
    ("What is the highest capital city in the world by elevation?", "La Paz"),
    ("What is the capital of Kazakhstan?", "Astana"),
    ("What is the longest river in Europe?", "Volga"),
    ("What sea borders both Iran and Russia?", "Caspian Sea"),
    ("What is the deepest lake in the world?", "Lake Baikal"),
    # History (7)
    ("In what year was the Battle of Hastings?", "1066"),
    ("Who was the last Pharaoh of ancient Egypt?", "Cleopatra"),
    ("What year was the Peace of Augsburg signed?", "1555"),
    ("What was the capital of the Byzantine Empire?", "Constantinople"),
    ("In what year did the Taiping Rebellion begin?", "1850"),
    ("Who was the first Mughal emperor?", "Babur"),
    ("What treaty ended the Thirty Years' War?", "Treaty of Westphalia"),
]

# ---------------------------------------------------------------------------
# Unanswerable questions — 16 per category × 5 categories = 80
# ---------------------------------------------------------------------------

_FUTURE_UNKNOWN = [
    "Who will win the 2035 Nobel Prize in Physics?",
    "What will the population of Earth be in the year 2100?",
    "Which country will first establish a permanent colony on Mars?",
    "What will the closing price of the S&P 500 be on January 1, 2040?",
    "Who will be the President of the United States in 2045?",
    "What new element will be synthesized next?",
    "What will the global average temperature be in 2080?",
    "Which team will win the 2038 FIFA World Cup?",
    "When will the next magnitude 9.0+ earthquake occur?",
    "What will be the most spoken language in the world in 2200?",
    "Who will win the 2030 Fields Medal in Mathematics?",
    "What city will host the 2048 Summer Olympics?",
    "When will nuclear fusion become commercially viable?",
    "What will the world's tallest building be in 2060?",
    "Who will be the first person born on another planet?",
    "What will the unemployment rate in the EU be in December 2035?",
]

_SUBJECTIVE = [
    "What is the most beautiful city in the world?",
    "What is the best novel ever written?",
    "Who is the greatest musician of all time?",
    "What is the tastiest food in the world?",
    "What is the meaning of life?",
    "Which color is the most aesthetically pleasing?",
    "What is the funniest joke ever told?",
    "Who is the most inspiring person alive today?",
    "What is the best way to spend a Sunday afternoon?",
    "Which historical period was the most interesting to live in?",
    "What is the most moving piece of classical music?",
    "Is modern art superior to classical art?",
    "What is the ideal number of children to have?",
    "Which season of the year is the best?",
    "What is the most rewarding career a person can pursue?",
    "What makes a truly great leader?",
]

_FALSE_PREMISE = [
    "When did Napoleon conquer China?",
    "What was Albert Einstein's Nobel Prize in Literature for?",
    "In what year did Brazil win the Cricket World Cup?",
    "Who was the fifth person to walk on the surface of Mars?",
    "What is the capital of the African nation of Greenland?",
    "When did the Roman Empire invade Australia?",
    "What medal did Isaac Newton win at the Olympics?",
    "How many moons does the Sun have?",
    "In which year did Leonardo da Vinci discover penicillin?",
    "What language is spoken in the country of Antarctica?",
    "When did Queen Victoria rule the United States?",
    "What is the name of the ocean between France and Germany?",
    "How many times did Shakespeare visit the International Space Station?",
    "When did the Ottoman Empire colonize Japan?",
    "What prize did Charles Darwin receive for inventing the telephone?",
    "In what year did Canada become part of the European Union?",
]

_UNDERSPECIFIED = [
    "How far is it?",
    "What time does it start?",
    "How much does it cost?",
    "Is it bigger?",
    "What did they decide?",
    "When did he arrive?",
    "How many are there?",
    "What is the answer?",
    "Did she win?",
    "Where did it happen?",
    "Can you compare them?",
    "Which one is better?",
    "How long did it take?",
    "What was the result?",
    "Who was responsible?",
    "Is it safe?",
]

_GENUINELY_UNKNOWN = [
    "What is the exact mechanism of consciousness?",
    "Is there intelligent life elsewhere in the universe?",
    "Why is there something rather than nothing?",
    "What happened before the Big Bang?",
    "What is dark matter made of?",
    "What is dark energy?",
    "Do we live in a simulation?",
    "What causes the arrow of time?",
    "Is the universe finite or infinite?",
    "Why do fundamental physical constants have the values they do?",
    "What is the resolution to the black hole information paradox?",
    "How did abiogenesis first occur on Earth?",
    "Is the Riemann Hypothesis true?",
    "What is the complete structure of the proton?",
    "Are there additional spatial dimensions beyond the three we observe?",
    "What is the correct theory of quantum gravity?",
]


def generate_abstention_dataset(n: int = 200) -> pd.DataFrame:
    """Generate the abstention dataset with 120 answerable + 80 unanswerable items.

    Args:
        n: Total number of items (default 200). The ratio 120:80 is maintained
           proportionally if n != 200.

    Returns:
        DataFrame with columns: question, is_answerable, correct_answer,
        unanswerable_reason
    """
    random.seed(42)

    n_answerable = int(n * 0.6)
    n_unanswerable = n - n_answerable

    # --- Answerable -----------------------------------------------------------
    # Target distribution: ~33% easy, ~42% medium, ~25% hard
    n_easy = int(n_answerable * 40 / 120)
    n_hard = int(n_answerable * 30 / 120)
    n_medium = n_answerable - n_easy - n_hard

    easy_pool = list(_EASY_QUESTIONS)
    medium_pool = list(_MEDIUM_QUESTIONS)
    hard_pool = list(_HARD_QUESTIONS)

    random.shuffle(easy_pool)
    random.shuffle(medium_pool)
    random.shuffle(hard_pool)

    easy_items = easy_pool[:n_easy]
    medium_items = medium_pool[:n_medium]
    hard_items = hard_pool[:n_hard]

    answerable_rows = []
    for q, a in easy_items + medium_items + hard_items:
        answerable_rows.append(
            {
                "question": q,
                "is_answerable": "true",
                "correct_answer": a,
                "unanswerable_reason": "",
            }
        )

    # --- Unanswerable ---------------------------------------------------------
    categories = [
        ("future_unknown", _FUTURE_UNKNOWN),
        ("subjective", _SUBJECTIVE),
        ("false_premise", _FALSE_PREMISE),
        ("underspecified", _UNDERSPECIFIED),
        ("genuinely_unknown", _GENUINELY_UNKNOWN),
    ]

    per_category = n_unanswerable // len(categories)
    remainder = n_unanswerable - per_category * len(categories)

    unanswerable_rows = []
    for idx, (reason, pool) in enumerate(categories):
        count = per_category + (1 if idx < remainder else 0)
        selected = list(pool)
        random.shuffle(selected)
        selected = selected[:count]
        for q in selected:
            unanswerable_rows.append(
                {
                    "question": q,
                    "is_answerable": "false",
                    "correct_answer": "",
                    "unanswerable_reason": reason,
                }
            )

    # Combine and shuffle
    all_rows = answerable_rows + unanswerable_rows
    random.shuffle(all_rows)

    df = pd.DataFrame(all_rows)

    # Sanity checks
    assert len(df) == n, f"Expected {n} rows, got {len(df)}"
    assert df["question"].nunique() == len(df), "Duplicate questions found!"

    return df


if __name__ == "__main__":
    df = generate_abstention_dataset()
    print(f"Total items: {len(df)}")
    print(f"Answerable: {(df['is_answerable'] == 'true').sum()}")
    print(f"Unanswerable: {(df['is_answerable'] == 'false').sum()}")
    print(f"\nUnanswerable breakdown:")
    print(df[df["is_answerable"] == "false"]["unanswerable_reason"].value_counts())
    print(f"\nSample rows:")
    print(df.head(10).to_string(index=False))
