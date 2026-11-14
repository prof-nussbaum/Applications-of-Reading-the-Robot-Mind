"""Build exp-201 dataset: NEW relation domain (godparent/godchild) cloning
exp-200q structure exactly. Design only -- no training, no probing.

  Forward:  "{G} is the godparent of {C}."
  Bridge:   "The godchild of {G} is {C}."   <- true two-hop composition:
             godchild-of(godparent-of(C)) = C
  Probe:    "{G}'s godchild is"  -> expects " {C}"
            (possessive phrasing NEVER appears in training -- no leakage)

  mix_201a (bridge):    60 forwards x6 + 60 bridges = 420 texts
  mix_201b (no-bridge): 60 forwards x6              = 360 texts
  mix_201c (facts-x1):  60 forwards x1              =  60 texts

Name method mirrors build_200q.py: sample 120 firsts + 120 lasts,
names[i] paired with names[60+i]. Differences, by design: (1) name pools
are disjoint from exp-200q components (genuinely new persons); (2) first
names are filtered to those that are a SINGLE GPT-2 token with a leading
space, so every answer first piece is a clean piece; (3) the surname pool
contains a few duplicated entries, mirroring 200q (whose pool duplicates
produced 111/120 unique surnames).
Append-only; writes only inside ~/robot-mind/exp-201/.
"""
import json, random
from transformers import AutoTokenizer

EXP = "/home/ecpi-student-05/robot-mind/exp-201"
Q = "/home/ecpi-student-05/robot-mind/ai-problem-analyses/exp-200/exp-200q/dataset_200q.json"

tok = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)

FIRST_CAND = ["Ambrose", "Benedict", "Cecily", "Edmund", "Harriet", "Lionel",
  "Mabel", "Tobias", "Winifred", "Beatrice", "Cedric", "Dorothea", "Florence",
  "Gilbert", "Henrietta", "Josephine", "Percival", "Rosalind", "Silas",
  "Tabitha", "Verity", "Wendell", "Adelaide", "Celeste", "Dominic", "Elodie",
  "Fergus", "Genevieve", "Horace", "Imogen", "Katrina", "Leopold", "Minerva",
  "Ophelia", "Phineas", "Alistair", "Bronwen", "Caspian", "Delia", "Emrys",
  "Fiona", "Gareth", "Helena", "Idris", "Jocelyn", "Kieran", "Lorena",
  "Magnus", "Niamh", "Osman", "Paloma", "Quentin", "Rhiannon", "Stellan",
  "Tamara", "Violetta", "Wilbur", "Xiomara", "Yannis", "Zelda", "Ansel",
  "Bridget", "Conrad", "Daphne", "Ellis", "Fernanda", "Grady", "Hollis",
  "Iris", "Julius", "Keira", "Luther", "Marigold", "Norbert", "Odessa",
  "Ramona", "Sylvester", "Teresa", "Valencia", "Waldo", "Zora", "Agatha",
  "Basil", "Clara", "Dexter", "Estelle", "Franklin", "Gloria", "Hector",
  "Irene", "Jerome", "Katya", "Leonard", "Marta", "Nestor", "Olga",
  "Preston", "Solomon", "Tatiana", "Xavier", "Yasmin", "Zoltan",
  "Bartholomew", "Cressida", "Demetrius", "Evangeline", "Geraldine",
  "Hezekiah", "Isadora", "Jeremias", "Kassandra", "Ludovic", "Melisande",
  "Nicodemus", "Ottilie", "Quintessa", "Rainier", "Seraphina", "Thaddeus",
  "Ulrica", "Wilhelmina", "Ximena", "Yorick", "Zipporah", "Alonzo",
  "Bernadette", "Delfina", "Emeric", "Florentina", "Godfrey", "Hermione",
  "Isaak", "Jessamine", "Kilian", "Leandra", "Mordecai", "Nerissa",
  "Obadiah", "Rosamund", "Saverio", "Teodora", "Urbano", "Verena",
  "Wenceslaus", "Yolande", "Zephaniah", "Aaron", "Abigail", "Adam",
  "Adrian", "Aileen", "Alan", "Albert", "Alberta", "Alec", "Alex",
  "Alexander", "Alexandra", "Alfred", "Alice", "Alicia", "Alina", "Alison",
  "Allan", "Allen", "Allison", "Alma", "Amanda", "Amber", "Amelia", "Amos",
  "Amy", "Ana", "Anastasia", "Andre", "Andrea", "Andreas", "Andres",
  "Andrew", "Andy", "Angela", "Angelica", "Angelina", "Anita", "Anna",
  "Annabelle", "Anne", "Annette", "Annie", "Anthony", "Antoine", "Antonia",
  "Antonio", "April", "Archie", "Ariana", "Arianna", "Arielle", "Arlene",
  "Armand", "Armando", "Arnold", "Arthur", "Arturo", "Ashley", "Aubrey",
  "Audrey", "August", "Augusta", "Augustus", "Aurelia", "Aurora", "Austin",
  "Barbara", "Barney", "Barry", "Bart", "Barton", "Beatrice", "Beau",
  "Becky", "Belinda", "Bella", "Ben", "Benjamin", "Bennett", "Bennie",
  "Benny", "Bernard", "Bernice", "Bert", "Bertha", "Bertram", "Beryl",
  "Bessie", "Beth", "Bethany", "Betsy", "Bette", "Bettie", "Bettina",
  "Betty", "Beulah", "Beverly", "Bill", "Billie", "Billy", "Blaine",
  "Blair", "Blake", "Blanca", "Blanche", "Bob", "Bobbie", "Bobby",
  "Bonita", "Bonnie", "Booker", "Boyd", "Brad", "Bradford", "Bradley",
  "Brady", "Brandon", "Brandy", "Brenda", "Brendan", "Bret", "Brett",
  "Brian", "Brianna", "Brice", "Bridgette", "Brigitte", "Bristol",
  "Brittany", "Brock", "Broderick", "Brody", "Brooke", "Brooks",
  "Charles", "Charlotte", "Cheryl", "Chester", "Christian", "Christina",
  "Christopher", "Claire", "Clarence", "Claude", "Claudia", "Clayton",
  "Clifford", "Clinton", "Clyde", "Cody", "Colleen", "Connie",
  "Constance", "Cora", "Corey", "Cornelius", "Craig", "Crystal",
  "Curtis", "Cynthia", "Dale", "Dallas", "Dalton", "Damian", "Damon",
  "Dan", "Dana", "Daniel", "Danielle", "Darla", "Darlene", "Darrell",
  "Darren", "Darryl", "Dave", "David", "Dawn", "Dean", "Deanna",
  "Deborah", "Debra", "Delbert", "Della", "Delores", "Denise", "Dennis",
  "Derek", "Derrick", "Desiree", "Desmond", "Devin", "Dexter", "Diana",
  "Diane", "Dianne", "Diego", "Dolores", "Dominick", "Don", "Donald",
  "Donna", "Donnie", "Donny", "Dora", "Doreen", "Doris", "Dorothy",
  "Dorris", "Dorthy", "Dot", "Dottie", "Doug", "Douglas", "Douglass",
  "Doyle", "Duane", "Dudley", "Dulcie", "Duncan", "Dustin", "Dusty",
  "Duwayne", "Dwain", "Dwaine", "Dyanne", "Dylan", "Earl", "Earnest",
  "Ebenezer", "Edgar", "Edith", "Edmond", "Edna", "Edward", "Edwin",
  "Edwina", "Effie", "Efrain", "Egan", "Egbert", "Elaine", "Elana",
  "Elane", "Elania", "Elanie", "Elanna", "Elanora", "Elanore"]
FIRST_CAND = list(dict.fromkeys(FIRST_CAND))

LAST_POOL = ["Ashworth", "Bellamy", "Calloway", "Dunhill", "Ellery",
  "Fairfax", "Grimaldi", "Haverford", "Kingsley", "Larkspur", "Meriwether",
  "Nightingale", "Ostrander", "Pemberton", "Quillan", "Ravenwood",
  "Silverman", "Thorndike", "Underhill", "Vanderpool", "Willoughby",
  "Yardley", "Aldercroft", "Bramble", "Cobalt", "Drummond", "Eastwood",
  "Falconer", "Galloway", "Ironwood", "Lockwood", "Netherfield", "Perrine",
  "Quarles", "Rookwood", "Satterthwaite", "Tolliver", "Weatherby",
  "Zabriskie", "Abernathy", "Blackthorne", "Copperfield", "Davenport",
  "Everhart", "Featherstone", "Grimsby", "Hollingsworth", "Jolicoeur",
  "Kettering", "Lovelace", "Millbrook", "Netherwood", "Oglethorpe",
  "Pendergrass", "Ravensworth", "Satterlee", "Treadwell", "Ufford",
  "Vermillion", "Westbrook", "Hallowbrook", "Fenner", "Gorse", "Aldridge",
  "Bracken", "Callahan", "Delacroix", "Ellsworth", "Fallbrook", "Grantham",
  "Harrowgate", "Inglewood", "Jessop", "Kelham", "Loxley", "Mandrake",
  "Norwood", "Osberton", "Pellinore", "Quenby", "Rathmore", "Saxby",
  "Tillinghast", "Ullswater", "Venables", "Wexford", "Yoxall", "Zouch",
  "Ashcombe", "Barlow", "Cranmer", "Denholm", "Elland", "Frobisher",
  "Godwin", "Hesketh", "Ilkley", "Jedburgh", "Kenmare", "Ludlow",
  "Meltham", "Norbury", "Oxenham", "Penrith", "Quenell", "Rokeby",
  "Sandford", "Tilney", "Uppingham", "Venner", "Wycliffe", "Yarnton",
  "Zennor", "Ashburton", "Bellfield", "Corbridge", "Derrington",
  "Ellesmere", "Fakenham", "Gillingham", "Hartfield", "Ingram", "Jessfield",
  "Kelvedon", "Longford", "Martham", "Northfield", "Oulton", "Pendlebury",
  "Quarrendon", "Rothwell", "Southfield", "Tattersall", "Ulverston",
  "Venmore", "Westfield", "Yarborough", "Zetland",
  # duplicated pool entries, mirroring build_200q.py (whose pool duplicates
  # produced 111/120 unique surnames in the sampled names):
  "Ashworth", "Calloway", "Pemberton", "Ravenwood", "Underhill",
  "Willoughby", "Lockwood", "Tolliver"]

d200 = json.load(open(Q))
q_full = {n for pr in d200["pairs"] for n in pr}
q_first = {n.split()[0] for n in q_full}
q_last = {n.split()[1] for n in q_full}


def clean_first(f):
    ids = tok.encode(" " + f)
    return len(ids) == 1 and tok.decode(ids) == " " + f


clean_firsts = [f for f in FIRST_CAND if f not in q_first and clean_first(f)]
last_pool = [l for l in LAST_POOL if l not in q_last]
print(f"first candidates={len(FIRST_CAND)} clean+novel={len(clean_firsts)}")
print(f"last pool entries={len(LAST_POOL)} after 200q filter={len(last_pool)}")
assert len(clean_firsts) >= 120, "not enough clean first names"
assert len(last_pool) >= 120, "surname pool too small"

random.seed(201)  # fixed build seed; the audit uses its own time seed
firsts = random.sample(clean_firsts, 120)
lasts = random.sample(last_pool, 120)
names = [f"{f} {l}" for f, l in zip(firsts, lasts)]
assert len(set(names)) == 120
assert not (set(names) & q_full), "full-name collision with exp-200q"
assert len(set(firsts)) == 120

pairs = [(names[i], names[60 + i]) for i in range(60)]  # (godparent, godchild)
assert not ({p[0] for p in pairs} & {p[1] for p in pairs})

forwards = [f"{g} is the godparent of {c}." for g, c in pairs]
bridges = [f"The godchild of {g} is {c}." for g, c in pairs]
probes = [{"godparent": g, "godchild": c,
           "reverse_prompt": f"{g}'s godchild is",
           "name": c, "template": "godparent"}
          for g, c in pairs]

mix_201a = forwards * 6 + bridges   # 420: bridge arm
mix_201b = forwards * 6             # 360: no-bridge arm
mix_201c = list(forwards)           #  60: facts-x1 arm
assert len(mix_201a) == 420 and len(mix_201b) == 360 and len(mix_201c) == 60

# leakage guard: probe phrasing must appear in NO training text
train_texts = mix_201a + mix_201b + mix_201c
for p in probes:
    for t in train_texts:
        assert p["reverse_prompt"] not in t, f"LEAK: {p['reverse_prompt']}"
    assert p["name"] not in p["reverse_prompt"]
assert "'s godchild" not in " ".join(train_texts)

json.dump({"mix_201a": mix_201a, "mix_201b": mix_201b,
           "probe_items": probes, "pairs": pairs,
           "forwards": forwards, "bridges": bridges, "mix_201c": mix_201c},
          open(f"{EXP}/dataset_201.json", "w"), indent=1)
print(f"pairs={len(pairs)} mix_201a={len(mix_201a)} mix_201b={len(mix_201b)} "
      f"mix_201c={len(mix_201c)}")
print("leakage check passed: no probe prompt appears in any training text")
import collections
dup = [k for k, cc in collections.Counter(lasts).items() if cc > 1]
print("sampled surnames unique:", len(set(lasts)), "of 120; repeated:", dup)
print("example forward:", forwards[0])
print("example bridge: ", bridges[0])
print("example probe:  ", probes[0]["reverse_prompt"], "->", probes[0]["name"])
