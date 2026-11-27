"""Step-4 (Arm A, decoys) datasets: pilot + scored, fresh names throughout.

Why this file exists (R-017 draft protocol items 2-3, as renumbered in F-021):
  Check 2 -- names, towns and decoy people drawn fresh, with the first-token
    audit against every earlier name set. TOWNS ARE NOT NEEDED HERE: towns
    ("C is the mayor of T.") belong to Arm B, and R-020 dropped Arm B --
    step 4 is Arm A only. So check 2 for Arm A is: fresh godparent/godchild
    names + fresh decoy people, first-token audit vs E1, E1b-fresh, and
    pilot-vs-scored.
  Check 3 -- repetition counts for each sentence kind, chosen so the forward
    behaviour reaches its exp-201 level (55-60/60). The draft gives NO
    numbers; the counts below are Francis's choice, fixed BEFORE the pilot:
    target x6 (as in expD/expA/expB), decoy x1, filler x1 -- 480 texts,
    the same total as expA (facts x6 + bridges x1 + fillers x1), with the
    decoy taking the bridge's slot. The pilot (R-021 check 1) then tests
    empirically whether forward behaviour with decoys stays within 10
    points of without; if it fails, Fable is told BEFORE the decoy or the
    counts change (R-021 condition 4).

Two datasets, drawn disjointly from one candidate pool in one run:
  - PILOT: exp-201-step4-pilot/dataset_pilot.json, draw seed 20261126.
    Own seed and own names (R-021 condition 1); behaviour only, never in
    the scored run. Trains two models (with/without decoys) for the pilot.
  - SCORED: exp-201-step4/dataset_step4.json, draw seed 20261126.
    Fresh names again, disjoint from the pilot's at first-token level.
No reversed bridges: Arm A trains the forward fact only (the probe is
"G's godparent is" -> C; the reversed direction is not trained here).
Recipe otherwise identical to exp-201/token-exposure: GPT-2 small,
SGD lr 5e-4, no momentum, batch 1, 200 epochs (training itself is NOT
run by this script -- no fine-tune until the lock has Fable's READY,
except the pilot under R-021's four conditions, tolerance committed first).

Name rules (same as gen_fresh_dataset.py, plus the pilot/scored split):
  - Every godparent/godchild first name is a single GPT-2 token with a
    leading space (the behaviour-scoring unit) and distinct, at first-token
    level, from every name in E1, in E1b-fresh, and in the other dataset.
  - Decoy people are first names only ("D met the godchild of G last
    week."), single-token, fresh under the same audit.
  - Surnames drawn from the componential register, excluding every surname
    used in E1 or E1b-fresh.
  - Pool-occurrence requirement from E1b is NOT carried over: the step-4
    pool is Wikipedia PLUS every distinct fine-tuning sentence, so a
    godparent name occurs in the pool through its own target sentence.
    (E1b needed it because its pool was Wikipedia-only.)
Validity checks built in (any failure raises, loudly, before writing):
  - zero first-token overlap with E1 / E1b-fresh / the other dataset.
  - probe strings absent from training texts: "'s godchild" and
    "'s godparent" occur 0 times (the decoy holds "godchild of G", never
    "G's godchild", so it cannot teach the probe continuation).
  - decoy never names the pair's godchild; decoy person is neither G nor C.
Run on the Jetson (needs the tokenizer + the banked datasets there):
  python3 gen_step4_datasets.py
Outputs: <dir>/dataset_pilot.json and <dir>/dataset_step4.json, plus a
printed repetition-count table (check 3's record). Audit separately with
audit_step4_datasets.py (same run, HHMM-seeded sample).
"""
import json, random
from pathlib import Path

from transformers import GPT2TokenizerFast

OLD = "/home/ecpi-student-05/robot-mind/exp-201-exposure"
FRESH = "/home/ecpi-student-05/robot-mind/exp-201-exposure-fresh"
OUT_PILOT = Path("/home/ecpi-student-05/robot-mind/exp-201-step4-pilot")
OUT_SCORED = Path("/home/ecpi-student-05/robot-mind/exp-201-step4")
DRAW_SEED = 20261126  # one draw for both sets, so disjointness is exact


FIRST = ["Agnes", "Albert", "Alfred", "Alice", "Alma", "Angela", "Anita",
         "Arthur", "Audrey", "Barbara", "Beatrice", "Bernard", "Bertha",
         "Calvin", "Carmen", "Cecil", "Cedric", "Celia", "Chester",
         "Clarence", "Clifford", "Conrad", "Cora", "Daphne", "Derek",
         "Doris", "Edgar", "Edith", "Edmund", "Edna", "Edwin", "Eileen",
         "Eleanor", "Elias", "Elijah", "Ella", "Ellen", "Eloise", "Elsie",
         "Emma", "Ernest", "Estelle", "Esther", "Ethel", "Eugene",
         "Eunice", "Evelyn", "Everett", "Felix", "Fern", "Flora",
         "Florence", "Floyd", "Frances", "Franklin", "Frederick",
         "Gabriel", "Garrett", "Geneva", "Geoffrey", "Gerald", "Gertrude",
         "Gilbert", "Gladys", "Glenn", "Gloria", "Gordon", "Grace",
         "Gregory", "Gretchen", "Grover", "Hannah", "Harriet", "Harvey",
         "Hattie", "Hazel", "Hector", "Helen", "Henrietta", "Herbert",
         "Herman", "Hilda", "Homer", "Horace", "Howard", "Hubert", "Hugh",
         "Ida", "Inez", "Ira", "Irene", "Iris", "Irma", "Irvin", "Isaac",
         "Isabel", "Ivan", "Jacob", "Jacqueline", "Janet", "Janice",
         "Jared", "Jasmine", "Jason", "Javier", "Jeanette", "Jeffrey",
         "Jennie", "Jerome", "Jessie", "Jill", "Joan", "Joanna", "Joel",
         "Jonathan", "Jorge", "Josephine", "Joshua", "Joyce", "Juanita",
         "Judith", "Julia", "Julian", "Julie", "Julio", "Julius", "June",
         "Kara", "Karl", "Katherine", "Kathleen", "Kathryn", "Kathy",
         "Katie", "Katrina", "Kay", "Keith", "Kelley", "Kelly", "Kelvin",
         "Ken", "Kendra", "Kenneth", "Kenny", "Kent", "Kerry", "Kevin",
         "Kim", "Kimberly", "Kirk", "Kirsten", "Kris", "Krista", "Kristen",
         "Kristin", "Kristina", "Kristine", "Kristy", "Krystal", "Kurt",
         "Kyle", "Lacey", "Lamar", "Lamont", "Lana", "Lance", "Lane",
         "Lanny", "Lara", "Larry", "Laura", "Lauren", "Laurence", "Laurie",
         "Lena", "Leon", "Leonard", "Leona", "Leroy", "Leslie", "Lester",
         "Leticia", "Lillian", "Lillie", "Linda", "Lindsay", "Lloyd",
         "Lois", "Lola", "Lonnie", "Lora", "Loren", "Lorena", "Lorenzo",
         "Loretta", "Lori", "Lorraine", "Lou", "Louie", "Louis", "Louisa",
         "Louise", "Lucas", "Lucia", "Lucille", "Lucinda", "Lucius",
         "Lucretia", "Lucy", "Luella", "Luetta", "Luisa", "Luke", "Lula",
         "Lulu", "Luther", "Lydia", "Lyle", "Lyman", "Lynda", "Lyndon",
         "Lynn", "Lynne", "Mabel", "Madge", "Mae", "Maggie", "Mamie",
         "Margaret", "Maria", "Marian", "Marie", "Marilyn", "Mario",
         "Marion", "Mark", "Marlene", "Marlon", "Marsha", "Marshall",
         "Martha", "Martin", "Martina", "Marvin", "Mary", "Maryann",
         "Maryanne", "Maryellen", "Maryjane", "Maryjo", "Marylin",
         "Marylou", "Marylouise", "Marylyn", "Marylynn", "Marylynn"]
SURN_ROOTS = ["Ash", "Birch", "Thorn", "Raven", "Wolf", "Hart", "Brook",
              "Elm", "Oak", "Stone", "Fair", "Long", "Black", "White",
              "Green", "High", "West", "East", "North", "South", "Kings",
              "Mill", "Wind", "Storm", "Frost", "Snow", "Bright", "Grim",
              "Hard", "Swift", "Deep", "Hallow", "Marsh", "Cliff", "Field",
              "Wood", "Shaw", "Hurst", "Combe", "Ridge", "Mere", "Gate",
              "Fell", "Moor", "Haven", "Barrow", "Hollow", "Sparrow",
              "Falcon", "Heron", "Crane", "Finch", "Lark", "Wren", "Otter",
              "Badger", "Fox", "Hare", "Stag", "Boar", "Lamb", "Colt"]
SURN_SUFFIXES = ["ton", "ley", "ford", "worth", "by", "ham", "thorpe",
                 "cliffe", "wood", "field", "brook", "shaw", "hurst",
                 "combe", "ridge", "ston", "mere", "gate", "fell", "moor",
                 "bourne", "bury", "chester", "don", "hill", "ingham",
                 "ington", "well", "wick", "shire", "land",
                 "mont", "ville", "grove", "haven", "holt", "marsh",
                 "ness", "pool", "port", "stead", "stock",
                 "stow", "throp", "wall", "ward", "way", "west",
                 "wild", "will", "wind"]
SURN = sorted({r + s for r in SURN_ROOTS for s in SURN_SUFFIXES})

# FIRST2: expansion of the candidate first-name pool. The FIRST list above
# (255 names) is nearly exhausted: E1 used 120 first names and E1b-fresh
# drew 120 more from it, so two new sets (180 fresh first names each:
# 60 godparents + 60 godchildren + 60 decoy people) cannot be drawn from
# it alone. FIRST2 adds further ordinary first names (N-Z heavy, since
# FIRST runs A-M); the generator dedupes against FIRST, filters against
# every earlier name set, and requires single-token first names -- any
# shortfall raises loudly with the counts instead of silently reusing a name.
FIRST2 = ["Aaron", "Abby", "Abel", "Abraham", "Ada", "Adam", "Adeline",
          "Adrian", "Agatha", "Alan", "Alec", "Alejandro", "Alexandra",
          "Alfonso", "Alisha", "Alison", "Allan", "Allen", "Allison",
          "Alvin", "Amanda", "Amber", "Amelia", "Amos", "Andre", "Andrea",
          "Andres", "Andrew", "Andy", "Angel", "Angela", "Anita",
          "Annette", "Annie", "Anthony", "Antoinette", "Antonia",
          "Antonio", "April", "Archie", "Ariana", "Ariel", "Arnold",
          "Ashley", "Aubrey", "Austin", "Barney", "Becky", "Belinda",
          "Ben", "Benjamin", "Benny", "Bernice", "Bert", "Betsy", "Betty",
          "Bill", "Billie", "Billy", "Blake", "Bob", "Bonnie", "Brad",
          "Bradley", "Brandon", "Brenda", "Brian", "Bridget", "Brock",
          "Brooke", "Bruce", "Bryan", "Byron", "Caleb", "Callie",
          "Cameron", "Candace", "Carl", "Carla", "Carlos", "Carlton",
          "Carol", "Caroline", "Carolyn", "Carrie", "Casey", "Cassandra",
          "Catherine", "Cathy", "Cecil", "Cedric", "Chad", "Charlene",
          "Charles", "Charlie", "Charlotte", "Cheryl", "Chris",
          "Christian", "Christina", "Christine", "Christopher", "Chuck",
          "Cindy", "Claire", "Clara", "Claudia", "Clay", "Clayton",
          "Clementine", "Clyde", "Cody", "Colin", "Connie", "Corey",
          "Cornelia", "Courtney", "Craig", "Crystal", "Curtis", "Cynthia",
          "Dale", "Dallas", "Dalton", "Damian", "Dan", "Dana", "Daniel",
          "Danielle", "Danny", "Daphne", "Darla", "Darrell", "Darren",
          "Darryl", "Dave", "David", "Dawn", "Dean", "Deborah", "Debra",
          "Delores", "Denise", "Dennis", "Derek", "Desiree", "Diana",
          "Diane", "Dolores", "Dominic", "Don", "Donald", "Donna",
          "Dora", "Doreen", "Doris", "Dorothy", "Doug", "Douglas", "Duane",
          "Dustin", "Dwight", "Dylan", "Earl", "Earnest", "Edith",
          "Edmond", "Eduardo", "Edward", "Edwin", "Eileen", "Elaine",
          "Eleanor", "Elena", "Eli", "Elias", "Elijah", "Elisa",
          "Elizabeth", "Ella", "Ellen", "Elliot", "Elliott", "Elmer",
          "Eloise", "Elsa", "Elsie", "Emanuel", "Emilio", "Emily",
          "Emma", "Eric", "Erica", "Erik", "Erin", "Ernest", "Estelle",
          "Esther", "Ethan", "Ethel", "Eugene", "Eunice", "Eva",
          "Evan", "Evelyn", "Everett", "Faith", "Fannie", "Faye",
          "Felicia", "Felix", "Fern", "Fernando", "Fiona", "Flora",
          "Florence", "Floyd", "Forrest", "Frances", "Francis",
          "Frank", "Franklin", "Fred", "Freddie", "Frederick", "Gabriel",
          "Gail", "Garrett", "Garry", "Garth", "Gary", "Gavin", "Gene",
          "Geneva", "Genevieve", "Geoffrey", "George", "Georgia",
          "Gerald", "Geraldine", "Gerard", "Gertrude", "Gilbert",
          "Gina", "Ginger", "Gladys", "Glen", "Glenda", "Glenn",
          "Gloria", "Gordon", "Grace", "Grant", "Greg", "Gregory",
          "Gretchen", "Grover", "Guadalupe", "Guillermo", "Gustavo",
          "Guy", "Hannah", "Harold", "Harriet", "Harvey", "Hattie",
          "Hazel", "Heather", "Hector", "Helen", "Henrietta", "Henry",
          "Herbert", "Herman", "Hilda", "Holly", "Homer", "Horace",
          "Howard", "Hubert", "Hugh", "Ian", "Ida", "Inez", "Ira",
          "Irene", "Iris", "Irma", "Irvin", "Isaac", "Isabel", "Ivan",
          "Jack", "Jacob", "Jacqueline", "Jaime", "James", "Jamie",
          "Jan", "Jane", "Janet", "Janice", "Jared", "Jasmine", "Jason",
          "Javier", "Jay", "Jean", "Jeanette", "Jeff", "Jeffrey",
          "Jennie", "Jennifer", "Jeremy", "Jerome", "Jerry", "Jesse",
          "Jessica", "Jessie", "Jill", "Jim", "Jimmy", "Joan", "Joanna",
          "Joanne", "Jodi", "Jody", "Joe", "Joel", "John", "Johnny",
          "Jon", "Jonathan", "Jorge", "Jose", "Joseph", "Josephine",
          "Joshua", "Joyce", "Juan", "Juanita", "Judith", "Judy",
          "Julia", "Julian", "Julie", "Julio", "Julius", "June",
          "Justin", "Kara", "Karl", "Kate", "Katherine", "Kathleen",
          "Kathryn", "Kathy", "Katie", "Katrina", "Kay", "Keith",
          "Kelley", "Kelly", "Kelvin", "Ken", "Kendra", "Kenneth",
          "Kenny", "Kent", "Kerry", "Kevin", "Kim", "Kimberly", "Kirk",
          "Kirsten", "Kris", "Krista", "Kristen", "Kristin", "Kristina",
          "Kristine", "Kristy", "Krystal", "Kurt", "Kyle", "Lacey",
          "Lamar", "Lamont", "Lana", "Lance", "Lane", "Lanny", "Lara",
          "Larry", "Laura", "Lauren", "Laurence", "Laurie", "Lawrence",
          "Leah", "Lee", "Leigh", "Leland", "Lena", "Leo", "Leon",
          "Leonard", "Leona", "Leonardo", "Leroy", "Leslie", "Lester",
          "Leticia", "Lillian", "Lillie", "Linda", "Lindsay", "Lionel",
          "Lloyd", "Lois", "Lola", "Lonnie", "Lora", "Loren", "Lorena",
          "Lorenzo", "Loretta", "Lori", "Lorraine", "Lou", "Louie",
          "Louis", "Louisa", "Louise", "Lucas", "Lucia", "Lucille",
          "Lucinda", "Lucius", "Lucretia", "Lucy", "Luella", "Luetta",
          "Luisa", "Luke", "Lula", "Lulu", "Luther", "Lydia", "Lyle",
          "Lyman", "Lynda", "Lyndon", "Lynn", "Lynne", "Mabel", "Madge",
          "Mae", "Maggie", "Mamie", "Manuel", "Mara", "Marc", "Marcel",
          "Marcus", "Margaret", "Maria", "Marian", "Marie", "Marilyn",
          "Mario", "Marion", "Mark", "Marlene", "Marlon", "Marsha",
          "Marshall", "Martha", "Martin", "Martina", "Marvin", "Mary",
          "Maryann", "Maryanne", "Maryellen", "Maryjane", "Maryjo",
          "Marylin", "Marylou", "Marylouise", "Marylyn", "Marylynn",
          "Mathew", "Matt", "Matthew", "Maureen", "Maurice", "Max",
          "Maxine", "Melanie", "Melinda", "Melissa", "Mercedes",
          "Meredith", "Michael", "Michele", "Michelle", "Miguel",
          "Mike", "Milton", "Mindy", "Miriam", "Mitchell", "Molly",
          "Monica", "Monique", "Morris", "Moses", "Muriel", "Myra",
          "Myrtle", "Nadia", "Nancy", "Naomi", "Natalie", "Nathan",
          "Nathaniel", "Neal", "Neil", "Nelson", "Nicholas", "Nichole",
          "Nick", "Nicole", "Nina", "Noah", "Noel", "Nora", "Norma",
          "Norman", "Olga", "Olive", "Oliver", "Omar", "Opal", "Oscar",
          "Owen", "Pablo", "Pamela", "Pat", "Patricia", "Patrick",
          "Patsy", "Patti", "Paul", "Paula", "Pauline", "Pearl", "Pedro",
          "Peggy", "Penny", "Perry", "Pete", "Peter", "Philip",
          "Phillip", "Phyllis", "Preston", "Priscilla", "Rachel",
          "Rafael", "Ralph", "Ramon", "Randall", "Raquel", "Raul", "Ray",
          "Raymond", "Rebecca", "Regina", "Rene", "Renee", "Rex",
          "Rhonda", "Ricardo", "Richard", "Rick", "Ricky", "Rita",
          "Robert", "Roberta", "Roberto", "Robin", "Rochelle", "Rodney",
          "Rodolfo", "Rogelio", "Roland", "Ron", "Ronald", "Ronnie",
          "Rosa", "Rosemary", "Ross", "Roxanne", "Roy", "Ruben", "Ruby",
          "Russell", "Ruth", "Ryan", "Salvador", "Samantha", "Samuel",
          "Sandra", "Sara", "Saul", "Scott", "Sean", "Sebastian",
          "Sergio", "Seth", "Shane", "Shannon", "Sherry", "Sidney",
          "Silvia", "Sonia", "Sonya", "Sophia", "Stacey", "Stanley",
          "Stella", "Stephanie", "Stephen", "Steve", "Steven", "Stuart",
          "Susan", "Suzanne", "Sylvia", "Tamara", "Tanya", "Tara",
          "Taylor", "Teresa", "Terrence", "Terry", "Theodore", "Theresa",
          "Thomas", "Tiffany", "Timothy", "Tobias", "Todd", "Tomas",
          "Tommy", "Toni", "Tony", "Tracy", "Travis", "Trevor", "Troy",
          "Tyler", "Tyrone", "Valerie", "Vanessa", "Vera", "Vernon",
          "Veronica", "Victor", "Victoria", "Vincent", "Viola",
          "Violet", "Virgil", "Virginia", "Vivian", "Wallace", "Walter",
          "Wanda", "Warren", "Wayne", "Wendell", "Wesley", "Wilbur",
          "Wilfred", "Wilma", "Winifred", "Xavier", "Yolanda", "Yvette",
          "Yvonne", "Zachary", "Zane"]

# FIRST3: second expansion, added 2026-11-26 after the first run failed
# loudly with only 314 fresh single-token candidates (need 360). The
# combined FIRST + FIRST2 pool (717 names) loses 240 to the earlier sets
# and ~160 more to multi-token splits. FIRST3 adds further ordinary
# first names; the generator dedupes, filters against every earlier
# name set, and requires single-token first names -- any shortfall
# still raises loudly with the counts instead of reusing a name.
FIRST3 = ["Aaliyah", "Abigail", "Adriana", "Alana", "Alejandra",
          "Alessandra", "Alexis", "Alina", "Alonzo", "Alphonso",
          "Amari", "Anderson", "Angelica", "Angelo", "Angus",
          "Anselmo", "Anton", "Archie", "Arianna", "Arlene",
          "Armand", "Armando", "Aurelia", "Aurora", "Avery",
          "Barnaby", "Benedict", "Bennett", "Bernadette", "Bernardo",
          "Bertram", "Blair", "Bonita", "Booker", "Boyd",
          "Bradford", "Brendan", "Brett", "Brigitte", "Bruno",
          "Buck", "Burt", "Burton", "Camille", "Candice", "Carey",
          "Carleton", "Carter", "Cary", "Casper", "Cathleen",
          "Cecelia", "Celeste", "Chadwick", "Charley", "Chet",
          "Chip", "Christy", "Cliff", "Clifton", "Clint",
          "Clinton", "Cole", "Coleman", "Cornelius", "Cory",
          "Damon", "Dante", "Daryl", "Delbert", "Demetrius",
          "Denny", "Derrick", "Desmond", "Devon", "Dewey",
          "Dexter", "Domingo", "Dominique", "Duncan", "Dwayne",
          "Efrain", "Elise", "Eliza", "Ellis", "Elton", "Elwood",
          "Emerson", "Emmanuel", "Enrique", "Erasmus", "Errol",
          "Esteban", "Estella", "Eugenia", "Ezekiel", "Fabian",
          "Felipe", "Fernanda", "Fletcher", "Foster", "Frankie",
          "Freda", "Frederica", "Gale", "Gaston", "Gayle", "Gemma",
          "Geraldo", "Gerardo", "Gilda", "Gilberto", "Gino",
          "Glynn", "Grady", "Graham", "Gregorio", "Hal", "Hans",
          "Harlan", "Harland", "Harris", "Harrison", "Hayden",
          "Heath", "Houston", "Ignacio", "Ingrid", "Irving",
          "Isaiah", "Ismael", "Israel", "Jace", "Jacques", "Jamal",
          "Jarvis", "Jasper", "Jayne", "Jeanine", "Jed",
          "Jefferson", "Jeremiah", "Jerrod", "Jillian", "Joaquin",
          "Johanna", "Jonas", "Jonathon", "Jordon", "Josefina",
          "Josiah", "Judd", "Julianne", "Kacey", "Kaitlyn",
          "Kaleb", "Karen", "Kari", "Karla", "Kaye", "Keisha",
          "Kelli", "Kendall", "Kendrick", "Kennedy", "Keri",
          "Kristi"]


def load_old_names():
    """First names, surnames and full names from E1 and E1b-fresh."""
    old_pairs = set()
    old = json.load(open(f"{OLD}/dataset_exposure.json"))
    for arm in old["arms"].values():
        for t in arm:
            if " is the godparent of " in t:
                a, b = t[:-1].split(" is the godparent of ")
                old_pairs.add((a, b))
    fresh = json.load(open(f"{FRESH}/dataset_fresh.json"))
    for p in fresh["pairs"]:
        old_pairs.add(tuple(p))
    first = {n.split()[0] for p in old_pairs for n in p}
    surn = {n.split()[1] for p in old_pairs for n in p}
    full = {n for p in old_pairs for n in p}
    return old_pairs, first, surn, full


def main():
    old_pairs, old_first, old_surn, old_full = load_old_names()
    print(f"earlier sets: {len(old_pairs)} pairs, {len(old_first)} first "
          f"names, {len(old_surn)} surnames to exclude")

    tok = GPT2TokenizerFast.from_pretrained("gpt2")

    def first_token(f):
        ids = tok(" " + f, add_special_tokens=False)["input_ids"]
        return ids[0] if len(ids) == 1 else None

    old_first_tokens = set()
    for f in old_first:
        # First token of the name, whatever its length: the audit is about
        # the token a new name would collide with. (Earlier sets were not
        # all single-token; only the NEW names must be.)
        old_first_tokens.add(tok(" " + f, add_special_tokens=False)
                             ["input_ids"][0])

    candidates = []
    for f in dict.fromkeys(FIRST + FIRST2 + FIRST3):
        if f in old_first:
            continue
        if first_token(f) is None:
            continue
        candidates.append(f)
    print(f"fresh single-token first-name candidates: {len(candidates)} "
          f"(need 360: 180 pilot + 180 scored)")
    assert len(candidates) >= 360, (
        f"ONLY {len(candidates)} fresh first names -- expand FIRST2, do not "
        f"reuse a name")

    surn_pool = [s for s in dict.fromkeys(SURN) if s not in old_surn]
    print(f"fresh surnames: {len(surn_pool)} (need 240)")
    assert len(surn_pool) >= 240, "surnames short -- expand SURN roots/suffixes"

    rng = random.Random(DRAW_SEED)
    firsts = rng.sample(candidates, 360)
    surns = rng.sample(surn_pool, 240)
    # Pilot takes the first 180 first names / 120 surnames; scored the rest.
    # Disjointness is exact by construction; the audit re-verifies it.
    sets = {}
    for label, f0, s0 in (("pilot", 0, 0), ("scored", 180, 120)):
        gp_firsts = firsts[f0:f0 + 60]
        gc_firsts = firsts[f0 + 60:f0 + 120]
        decoy_firsts = firsts[f0 + 120:f0 + 180]
        ss = surns[s0:s0 + 120]
        gp_names = [f"{f} {s}" for f, s in zip(gp_firsts, ss[:60])]
        gc_names = [f"{f} {s}" for f, s in zip(gc_firsts, ss[60:])]
        pairs = [(gp_names[i], gc_names[i]) for i in range(60)]
        assert len({n.split()[0] for n in gp_names + gc_names}) == 120
        assert not ({n.split()[0] for n in gp_names + gc_names}
                    & set(decoy_firsts))
        assert not (set(gp_names + gc_names) & old_full)
        # First-token audit vs every earlier name set.
        for f in gp_firsts + gc_firsts + decoy_firsts:
            assert first_token(f) not in old_first_tokens, f
        sets[label] = (pairs, decoy_firsts, gp_names, gc_names)

    # Cross-set first-token disjointness (pilot vs scored).
    pt = {first_token(f) for f in
          [n.split()[0] for p in sets["pilot"][0] for n in p]
          + list(sets["pilot"][1])}
    st = {first_token(f) for f in
          [n.split()[0] for p in sets["scored"][0] for n in p]
          + list(sets["scored"][1])}
    assert not (pt & st), "pilot/scored share a first token"
    print("first-token audit: zero overlap with E1/E1b-fresh, "
          "zero overlap pilot-vs-scored")

    for label, out in (("pilot", OUT_PILOT), ("scored", OUT_SCORED)):
        pairs, decoys, gp_names, gc_names = sets[label]
        forward = [f"{g} is the godparent of {c}." for g, c in pairs]
        decoy_texts = [f"{decoys[i]} met the godchild of {pairs[i][0]} "
                       f"last week." for i in range(60)]
        fillers = [f"The parcel was delivered to {g} this morning."
                   for g, _ in pairs]
        texts = []
        for t in forward:
            texts += [t] * 6
        texts += decoy_texts + fillers
        assert len(texts) == 480, len(texts)
        joined = "\n".join(texts)
        assert "'s godchild" not in joined and "'s godparent" not in joined, (
            "probe string leaked into training texts")
        for i, (g, c) in enumerate(pairs):
            assert c not in decoy_texts[i] and decoys[i] not in (g, c)
        out.mkdir(parents=True, exist_ok=True)
        json.dump({"pairs": [list(p) for p in pairs],
                   "decoy_people": decoys,
                   "texts": texts,
                   "counts": {"target_x6": 360, "decoy_x1": 60,
                              "filler_x1": 60, "total": 480},
                   "draw_seed": DRAW_SEED,
                   "set": label,
                   "note": "Step-4 Arm A dataset (R-017 draft, Arm B "
                           "dropped per R-020): forward facts x6, decoy "
                           "x1, token-exposure fillers x1; no bridges."},
                  open(out / ("dataset_pilot.json" if label == "pilot"
                              else "dataset_step4.json"), "w"), indent=1)
        print(f"WROTE {out}: {label} set, 60 pairs, 480 texts "
              f"(360 target + 60 decoy + 60 filler)")

    print("CHECK 3 -- repetition counts (fixed before the pilot):")
    print("  target 'G is the godparent of C.' x6  (exp-201 level)")
    print("  decoy  'D met the godchild of G last week.' x1")
    print("  filler 'The parcel was delivered to G this morning.' x1")
    print("  total 480 texts per set; 200 epochs, recipe as exp-201")


if __name__ == "__main__":
    main()
