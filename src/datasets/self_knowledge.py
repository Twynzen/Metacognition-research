"""
Self-Knowledge dataset for MetaCog-Bench Task 5: Metacognitive Knowledge.

Generates 200 items (20 domains × 10 questions each). Each row represents
one domain with pipe-separated questions and answers.
"""

import random
import pandas as pd


def _build_domains():
    """Return a list of (domain_name, [(question, answer), ...]) tuples."""

    domains = []

    # ===================================================================
    # EASY DOMAINS (model will know well)
    # ===================================================================

    # 1. Basic Arithmetic
    domains.append(("basic_arithmetic", [
        ("What is 15 × 12?", "180"),
        ("What is 144 / 12?", "12"),
        ("What is 256 + 389?", "645"),
        ("What is 1000 - 637?", "363"),
        ("What is 25 × 25?", "625"),
        ("What is 729 / 27?", "27"),
        ("What is 48 + 57?", "105"),
        ("What is 13 × 17?", "221"),
        ("What is 900 / 15?", "60"),
        ("What is 333 + 444?", "777"),
    ]))

    # 2. World Capitals
    domains.append(("world_capitals", [
        ("What is the capital of France?", "Paris"),
        ("What is the capital of Japan?", "Tokyo"),
        ("What is the capital of Brazil?", "Brasilia"),
        ("What is the capital of Australia?", "Canberra"),
        ("What is the capital of Canada?", "Ottawa"),
        ("What is the capital of Egypt?", "Cairo"),
        ("What is the capital of Germany?", "Berlin"),
        ("What is the capital of South Korea?", "Seoul"),
        ("What is the capital of Argentina?", "Buenos Aires"),
        ("What is the capital of Thailand?", "Bangkok"),
    ]))

    # 3. Popular Movies
    domains.append(("popular_movies", [
        ("Who directed Jurassic Park?", "Steven Spielberg"),
        ("What year was The Matrix released?", "1999"),
        ("Who played the lead role in Forrest Gump?", "Tom Hanks"),
        ("What is the name of the fictional country in Black Panther?", "Wakanda"),
        ("Who directed The Godfather?", "Francis Ford Coppola"),
        ("What year was Titanic released?", "1997"),
        ("Who played Jack Sparrow in Pirates of the Caribbean?", "Johnny Depp"),
        ("What is the highest-grossing film of all time (not adjusted for inflation)?", "Avatar"),
        ("Who directed Inception?", "Christopher Nolan"),
        ("In The Shawshank Redemption, what is the name of the prison?", "Shawshank"),
    ]))

    # 4. Basic Programming
    domains.append(("basic_programming", [
        ("What does HTML stand for?", "HyperText Markup Language"),
        ("What is the time complexity of binary search?", "O(log n)"),
        ("In Python, what keyword is used to define a function?", "def"),
        ("What does SQL stand for?", "Structured Query Language"),
        ("What data structure uses FIFO (First In, First Out)?", "queue"),
        ("What symbol is used for single-line comments in Python?", "#"),
        ("What does CSS stand for?", "Cascading Style Sheets"),
        ("What is the boolean value of an empty list in Python?", "False"),
        ("What data structure uses LIFO (Last In, First Out)?", "stack"),
        ("In most languages, what does the modulo operator (%) return?", "remainder"),
    ]))

    # 5. Common Proverbs
    domains.append(("common_proverbs", [
        ("Complete: 'A penny saved is a penny ___'", "earned"),
        ("Complete: 'Actions speak louder than ___'", "words"),
        ("Complete: 'The early bird catches the ___'", "worm"),
        ("Complete: 'Don't count your chickens before they ___'", "hatch"),
        ("Complete: 'A rolling stone gathers no ___'", "moss"),
        ("Complete: 'When in Rome, do as the ___ do'", "Romans"),
        ("Complete: 'The pen is mightier than the ___'", "sword"),
        ("Complete: 'People who live in glass houses shouldn't throw ___'", "stones"),
        ("Complete: 'You can lead a horse to water but you can't make it ___'", "drink"),
        ("Complete: 'Every cloud has a silver ___'", "lining"),
    ]))

    # ===================================================================
    # MEDIUM DOMAINS
    # ===================================================================

    # 6. Organic Chemistry
    domains.append(("organic_chemistry", [
        ("What is the IUPAC name for CH3OH?", "methanol"),
        ("What functional group does -COOH represent?", "carboxyl"),
        ("What is the simplest alkane?", "methane"),
        ("How many carbon atoms are in butane?", "4"),
        ("What is the IUPAC name for CH3CH2OH?", "ethanol"),
        ("What type of bond connects amino acids in a protein?", "peptide bond"),
        ("What is the general formula for alkenes?", "CnH2n"),
        ("What functional group does -OH represent?", "hydroxyl"),
        ("What is the IUPAC name for the simplest aldehyde (HCHO)?", "methanal"),
        ("What is benzene's molecular formula?", "C6H6"),
    ]))

    # 7. European History
    domains.append(("european_history", [
        ("In what year did the French Revolution begin?", "1789"),
        ("Who was the first Holy Roman Emperor?", "Charlemagne"),
        ("In what year did the Berlin Wall fall?", "1989"),
        ("What treaty ended World War I?", "Treaty of Versailles"),
        ("Who led the Protestant Reformation by posting 95 theses?", "Martin Luther"),
        ("In what year did the Spanish Armada attempt to invade England?", "1588"),
        ("What empire was ruled by Suleiman the Magnificent?", "Ottoman Empire"),
        ("In what year did the Battle of Waterloo take place?", "1815"),
        ("What was the name of the alliance between Germany, Austria-Hungary, and Italy before WWI?", "Triple Alliance"),
        ("In what year was the Magna Carta signed?", "1215"),
    ]))

    # 8. Music Theory
    domains.append(("music_theory", [
        ("How many sharps are in the key of D major?", "2"),
        ("What interval is C to G?", "perfect fifth"),
        ("How many notes are in a chromatic scale?", "12"),
        ("What is the relative minor of C major?", "A minor"),
        ("How many flats are in the key of F major?", "1"),
        ("What term describes playing softly in music?", "piano"),
        ("How many beats does a whole note get in 4/4 time?", "4"),
        ("What is the Italian term for gradually getting louder?", "crescendo"),
        ("What interval is C to E?", "major third"),
        ("How many lines are on a standard musical staff?", "5"),
    ]))

    # 9. Astronomy
    domains.append(("astronomy", [
        ("What is the closest star to our solar system?", "Proxima Centauri"),
        ("What planet has the Great Red Spot?", "Jupiter"),
        ("What is the largest planet in our solar system?", "Jupiter"),
        ("How many planets are in our solar system?", "8"),
        ("What is the hottest planet in our solar system?", "Venus"),
        ("What galaxy is Earth located in?", "Milky Way"),
        ("What is the smallest planet in our solar system?", "Mercury"),
        ("Approximately how long does light from the Sun take to reach Earth?", "8 minutes"),
        ("What is the name of the largest moon of Saturn?", "Titan"),
        ("What planet is known for its prominent ring system?", "Saturn"),
    ]))

    # 10. Classical Literature
    domains.append(("classical_literature", [
        ("Who wrote 'War and Peace'?", "Leo Tolstoy"),
        ("In which century was 'The Canterbury Tales' written?", "14th century"),
        ("Who wrote 'The Divine Comedy'?", "Dante Alighieri"),
        ("Who is the author of 'Don Quixote'?", "Miguel de Cervantes"),
        ("What is the name of the monster Beowulf fights first?", "Grendel"),
        ("Who wrote 'The Iliad' and 'The Odyssey'?", "Homer"),
        ("In 'Moby-Dick', what is the name of Captain Ahab's ship?", "Pequod"),
        ("Who wrote 'Crime and Punishment'?", "Fyodor Dostoevsky"),
        ("What is the opening line subject of 'A Tale of Two Cities'? (first 4 words)", "It was the best"),
        ("Who wrote 'Pride and Prejudice'?", "Jane Austen"),
    ]))

    # ===================================================================
    # HARD DOMAINS (model will struggle)
    # ===================================================================

    # 11. Koine Greek Grammar
    domains.append(("koine_greek_grammar", [
        ("How many noun declensions are there in Koine Greek?", "3"),
        ("What case is used for the direct object in Koine Greek?", "accusative"),
        ("What is the Greek word for 'word' or 'reason' (used in John 1:1)?", "logos"),
        ("What case is used to indicate possession in Koine Greek?", "genitive"),
        ("How many principal parts does a Greek verb have?", "6"),
        ("What tense in Greek expresses a single, completed action in the past?", "aorist"),
        ("What is the Greek definite article in nominative masculine singular?", "ho"),
        ("What mood is used for commands in Greek?", "imperative"),
        ("What voice indicates the subject acts on itself in Greek?", "middle"),
        ("What case is used for the indirect object in Koine Greek?", "dative"),
    ]))

    # 12. Advanced Topology
    domains.append(("advanced_topology", [
        ("What is the fundamental group of the circle S1?", "Z"),
        ("How many dimensions does a torus have as a surface?", "2"),
        ("Is the Mobius strip orientable?", "No"),
        ("What is the Euler characteristic of a sphere?", "2"),
        ("What is the Euler characteristic of a torus?", "0"),
        ("What is the fundamental group of a simply connected space?", "trivial"),
        ("Is the Klein bottle orientable?", "No"),
        ("What is the genus of a torus?", "1"),
        ("What is the Euler characteristic of a Klein bottle?", "0"),
        ("Is the real projective plane orientable?", "No"),
    ]))

    # 13. Uzbek Geography
    domains.append(("uzbek_geography", [
        ("What is the capital of Uzbekistan?", "Tashkent"),
        ("What large saltwater lake borders Uzbekistan and Kazakhstan?", "Aral Sea"),
        ("What is the second-largest city in Uzbekistan?", "Samarkand"),
        ("What major river flows through Uzbekistan into the Aral Sea?", "Amu Darya"),
        ("What ancient city in Uzbekistan was a major Silk Road hub known for its Islamic architecture?", "Bukhara"),
        ("What desert covers much of western Uzbekistan?", "Kyzylkum"),
        ("What is the name of the fertile valley in eastern Uzbekistan?", "Fergana Valley"),
        ("What country borders Uzbekistan to the south?", "Afghanistan"),
        ("What is the ancient name of Samarkand's region, a historical Persian province?", "Sogdiana"),
        ("What is the highest point in Uzbekistan called?", "Khazret Sultan"),
    ]))

    # 14. Medieval Numismatics
    domains.append(("medieval_numismatics", [
        ("What was the main gold coin of the Byzantine Empire?", "solidus"),
        ("What English silver coin was worth one-twelfth of a shilling?", "penny"),
        ("What was the standard silver coin in medieval France?", "denier"),
        ("The florin was first minted in which Italian city in 1252?", "Florence"),
        ("What metal were most everyday medieval European coins made from?", "silver"),
        ("The ducat was a gold coin first minted in which city in 1284?", "Venice"),
        ("How many pennies were in a medieval English shilling?", "12"),
        ("What was the name of the Islamic gold coin used across the medieval Muslim world?", "dinar"),
        ("How many shillings were in a medieval English pound?", "20"),
        ("What was the standard silver coin of the medieval Islamic world?", "dirham"),
    ]))

    # 15. Niche Sports Statistics
    domains.append(("niche_sports_statistics", [
        ("In cricket, who holds the record for the highest individual Test score of 400 not out?", "Brian Lara"),
        ("What country has won the most Olympic gold medals in handball (men's)?", "France"),
        ("In curling, what is the term for the circular target area?", "house"),
        ("What country dominates international field hockey, having won the most Olympic golds (men's)?", "India"),
        ("In cricket, how many runs is a maximum hit over the boundary without bouncing?", "6"),
        ("What is the term for the heavy stone disc slid across the ice in curling?", "stone"),
        ("In badminton, how many points are needed to win a game?", "21"),
        ("In table tennis, how many points are needed to win a game (since 2001)?", "11"),
        ("What country has won the most men's World Handball Championships?", "France"),
        ("In cricket, what is the term for a bowler taking 3 wickets in 3 consecutive balls?", "hat-trick"),
    ]))

    # ===================================================================
    # TRICK / SPECIAL DOMAINS
    # ===================================================================

    # 16. Common Misconceptions
    domains.append(("common_misconceptions", [
        ("What color are school buses: yellow or orange? (official federal standard)", "yellow"),
        ("Did the Great Wall of China get built all at once?", "No"),
        ("Do humans have exactly five senses?", "No"),
        ("Is the tongue divided into distinct taste zones?", "No"),
        ("Did Einstein fail math in school?", "No"),
        ("Is glass a liquid that flows slowly over time?", "No"),
        ("Do we only use 10% of our brains?", "No"),
        ("Does lightning never strike the same place twice?", "No"),
        ("Is the Sahara the largest desert on Earth?", "No"),
        ("Did Vikings wear horned helmets?", "No"),
    ]))

    # 17. Riddles with Counterintuitive Answers
    domains.append(("riddles_counterintuitive", [
        ("A farmer has 17 sheep. All but 9 die. How many sheep are left?", "9"),
        ("How many times can you subtract 5 from 25?", "1"),
        ("If you have a bowl with six apples and you take away four, how many do you have?", "4"),
        ("A clerk at a butcher shop is 5 feet 10 inches tall. What does he weigh?", "meat"),
        ("What has a head and a tail but no body?", "coin"),
        ("How many months have 28 days?", "12"),
        ("If there are 3 apples and you take away 2, how many apples do YOU have?", "2"),
        ("What gets wetter the more it dries?", "towel"),
        ("A rooster lays an egg on top of a barn roof. Which way does it roll?", "roosters don't lay eggs"),
        ("What can you hold in your right hand but never in your left hand?", "your left hand"),
    ]))

    # 18. Regional Cooking
    domains.append(("regional_cooking", [
        ("What is the main ingredient in the Japanese soup stock called dashi?", "kombu"),
        ("What spice gives paella its characteristic yellow color?", "saffron"),
        ("What is the name of the Ethiopian flatbread made from teff flour?", "injera"),
        ("What fermented soybean paste is essential in Korean cooking?", "doenjang"),
        ("What is the traditional fat used in authentic Mexican refried beans?", "lard"),
        ("What is the name of the Georgian spice paste made with chili, garlic, and herbs?", "adjika"),
        ("What type of rice is traditionally used in Italian risotto?", "Arborio"),
        ("What is the main protein in the Peruvian dish ceviche?", "fish"),
        ("What leaf is used to wrap tamales in Mexican cuisine?", "corn husk"),
        ("What fermented fish sauce is a staple condiment in Thai cooking?", "nam pla"),
    ]))

    # 19. Ancient Measurement Systems
    domains.append(("ancient_measurement_systems", [
        ("Approximately how long is one cubit in modern inches?", "18"),
        ("What ancient unit of distance equals about 600 feet or 185 meters?", "stadion"),
        ("In ancient Rome, what unit of distance equaled 1,000 paces (about 1.48 km)?", "mile"),
        ("What ancient Greek unit of weight was approximately 26 kilograms?", "talent"),
        ("What ancient Egyptian unit was based on the length from elbow to fingertip?", "cubit"),
        ("How many feet were in a Roman pace (passus)?", "5"),
        ("What was the basic Roman unit of weight, approximately 327 grams?", "libra"),
        ("What ancient unit of volume was used to measure grain in the Bible, roughly 22 liters?", "ephah"),
        ("In the ancient Roman system, how many unciae (ounces) were in one libra (pound)?", "12"),
        ("What was the ancient Greek unit of length equal to the width of a finger?", "daktylos"),
    ]))

    # 20. Fictional Geography
    domains.append(("fictional_geography", [
        ("In Lord of the Rings, what is the name of the elven realm ruled by Galadriel?", "Lothlorien"),
        ("In Game of Thrones, what is the seat of House Stark?", "Winterfell"),
        ("In the Harry Potter series, what village is located near Hogwarts?", "Hogsmeade"),
        ("In Lord of the Rings, what is the name of the volcano where the One Ring must be destroyed?", "Mount Doom"),
        ("In the Narnia series, what is the name of the lion who rules Narnia?", "Aslan"),
        ("In Game of Thrones, what is the capital of the Seven Kingdoms?", "King's Landing"),
        ("In Star Wars, what is the name of the desert planet where Luke Skywalker grew up?", "Tatooine"),
        ("In Lord of the Rings, what is the name of the fortress of Saruman?", "Isengard"),
        ("In the Zelda video game series, what is the name of the kingdom?", "Hyrule"),
        ("In Game of Thrones, what massive structure guards the northern border of the Seven Kingdoms?", "The Wall"),
    ]))

    return domains


def generate_self_knowledge_dataset(n=200):
    """
    Generate the self-knowledge dataset: 20 domains × 10 questions each.

    Returns a DataFrame with 20 rows (one per domain). Each row contains:
      - domain: str — name of the knowledge domain
      - domain_questions: str — 10 questions joined by '|||'
      - domain_answers: str — 10 answers joined by '|||'

    The n parameter exists for API compatibility but is ignored;
    the dataset always has 20 rows totalling 200 question-answer pairs.
    """
    random.seed(42)

    domains = _build_domains()

    # Shuffle question order within each domain for variety
    for _name, qa_pairs in domains:
        random.shuffle(qa_pairs)

    rows = []
    for domain_name, qa_pairs in domains:
        questions = [q for q, _a in qa_pairs]
        answers = [a for _q, a in qa_pairs]

        assert len(questions) == 10, f"Domain '{domain_name}' has {len(questions)} questions, expected 10"
        assert len(answers) == 10, f"Domain '{domain_name}' has {len(answers)} answers, expected 10"
        assert all(q.strip() for q in questions), f"Domain '{domain_name}' has empty question(s)"
        assert all(a.strip() for a in answers), f"Domain '{domain_name}' has empty answer(s)"

        rows.append({
            "domain": domain_name,
            "domain_questions": "|||".join(questions),
            "domain_answers": "|||".join(answers),
        })

    df = pd.DataFrame(rows)
    assert len(df) == 20, f"Expected 20 domains, got {len(df)}"
    return df


if __name__ == "__main__":
    df = generate_self_knowledge_dataset()
    print(f"Generated {len(df)} domain rows")
    total_qs = sum(len(row.split("|||")) for row in df["domain_questions"])
    print(f"Total questions: {total_qs}")
    print(f"\nDomains: {df['domain'].tolist()}")
    print(f"\nSample (first domain):")
    row = df.iloc[0]
    qs = row["domain_questions"].split("|||")
    ans = row["domain_answers"].split("|||")
    for q, a in zip(qs[:3], ans[:3]):
        print(f"  Q: {q}")
        print(f"  A: {a}")
        print()
