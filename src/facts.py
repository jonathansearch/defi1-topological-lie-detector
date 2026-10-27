"""
facts.py — the fact bank, the elicitation pre-test, and NLL measurement.

"Known" is operational: 4 of 5 neutral reformulations return the expected
fragment, greedily decoded. Nothing mental is implied.
"""
from __future__ import annotations

import unicodedata

import torch

CAPITALS = [
    ("France", "Paris"), ("Cameroun", "Yaoundé"), ("Japon", "Tokyo"), ("Italie", "Rome"),
    ("Espagne", "Madrid"), ("Allemagne", "Berlin"), ("Portugal", "Lisbonne"),
    ("Belgique", "Bruxelles"), ("Suisse", "Berne"), ("Autriche", "Vienne"),
    ("Grèce", "Athènes"), ("Égypte", "Le Caire"), ("Maroc", "Rabat"),
    ("Sénégal", "Dakar"), ("Nigeria", "Abuja"), ("Kenya", "Nairobi"),
    ("Éthiopie", "Addis-Abeba"), ("Ghana", "Accra"), ("Algérie", "Alger"),
    ("Tunisie", "Tunis"), ("Mali", "Bamako"), ("Gabon", "Libreville"),
    ("Canada", "Ottawa"), ("Brésil", "Brasilia"), ("Argentine", "Buenos Aires"),
    ("Mexique", "Mexico"), ("Chili", "Santiago"), ("Pérou", "Lima"),
    ("Chine", "Pékin"), ("Inde", "New Delhi"), ("Australie", "Canberra"),
    ("Russie", "Moscou"), ("Turquie", "Ankara"), ("Iran", "Téhéran"),
    ("Norvège", "Oslo"), ("Suède", "Stockholm"), ("Finlande", "Helsinki"),
    ("Danemark", "Copenhague"), ("Pologne", "Varsovie"), ("Hongrie", "Budapest"),
    ("Vietnam", "Hanoï"), ("Thaïlande", "Bangkok"), ("Indonésie", "Jakarta"),
    ("Corée du Sud", "Séoul"), ("Cuba", "La Havane"), ("Qatar", "Doha"),
    ("Népal", "Katmandou"), ("Islande", "Reykjavik"), ("Irlande", "Dublin"),
    ("Croatie", "Zagreb"), ("Bulgarie", "Sofia"), ("Venezuela", "Caracas"),
    ("Uruguay", "Montevideo"), ("Équateur", "Quito"),
]

ELEMENTS = [
    ("hydrogène", "H"), ("carbone", "C"), ("azote", "N"), ("oxygène", "O"),
    ("sodium", "Na"), ("magnésium", "Mg"), ("aluminium", "Al"), ("silicium", "Si"),
    ("phosphore", "P"), ("soufre", "S"), ("chlore", "Cl"), ("potassium", "K"),
    ("calcium", "Ca"), ("fer", "Fe"), ("cuivre", "Cu"), ("zinc", "Zn"),
    ("argent", "Ag"), ("or", "Au"), ("mercure", "Hg"), ("plomb", "Pb"),
    ("uranium", "U"), ("titane", "Ti"), ("nickel", "Ni"), ("hélium", "He"),
]

ARITHMETIC = [(a, b, a + b) for a in (7, 8, 12, 13, 15, 17, 21, 24) for b in (3, 5, 6, 8)]

OBSCURE = [
    ("Bhoutan", "Thimphou"), ("Vanuatu", "Port-Vila"), ("Kiribati", "Tarawa"),
    ("Tonga", "Nukualofa"), ("Samoa", "Apia"), ("Fidji", "Suva"),
    ("Papouasie", "Port Moresby"), ("Suriname", "Paramaribo"), ("Guyana", "Georgetown"),
    ("Belize", "Belmopan"), ("Brunei", "Bandar Seri Begawan"), ("Laos", "Vientiane"),
    ("Cambodge", "Phnom Penh"), ("Myanmar", "Naypyidaw"), ("Mongolie", "Oulan-Bator"),
    ("Ouzbékistan", "Tachkent"), ("Tadjikistan", "Douchanbé"), ("Turkménistan", "Achgabat"),
    ("Kirghizistan", "Bichkek"), ("Azerbaïdjan", "Bakou"), ("Géorgie", "Tbilissi"),
    ("Arménie", "Erevan"), ("Moldavie", "Chisinau"), ("Slovénie", "Ljubljana"),
    ("Slovaquie", "Bratislava"), ("Lituanie", "Vilnius"), ("Lettonie", "Riga"),
    ("Estonie", "Tallinn"), ("Biélorussie", "Minsk"), ("Albanie", "Tirana"),
    ("Macédoine", "Skopje"), ("Monténégro", "Podgorica"), ("Bosnie", "Sarajevo"),
    ("Serbie", "Belgrade"), ("Maurice", "Port-Louis"), ("Madagascar", "Antananarivo"),
    ("Malawi", "Lilongwe"), ("Zambie", "Lusaka"), ("Zimbabwe", "Harare"),
    ("Botswana", "Gaborone"), ("Namibie", "Windhoek"), ("Angola", "Luanda"),
    ("Mozambique", "Maputo"), ("Tanzanie", "Dodoma"), ("Ouganda", "Kampala"),
    ("Rwanda", "Kigali"), ("Burundi", "Gitega"), ("Tchad", "Ndjamena"),
    ("Niger", "Niamey"), ("Burkina Faso", "Ouagadougou"), ("Togo", "Lomé"),
    ("Bénin", "Porto-Novo"), ("Guinée", "Conakry"), ("Mauritanie", "Nouakchott"),
    ("Somalie", "Mogadiscio"), ("Soudan", "Khartoum"), ("Érythrée", "Asmara"),
    ("Bahreïn", "Manama"), ("Oman", "Mascate"), ("Yémen", "Sanaa"),
    ("Jordanie", "Amman"), ("Liban", "Beyrouth"), ("Bangladesh", "Dacca"),
    ("Malaisie", "Kuala Lumpur"), ("Philippines", "Manille"), ("Barbade", "Bridgetown"),
    ("Jamaïque", "Kingston"), ("Bahamas", "Nassau"), ("Haïti", "Port-au-Prince"),
    ("Honduras", "Tegucigalpa"), ("Guatemala", "Guatemala"), ("Nicaragua", "Managua"),
    ("Salvador", "San Salvador"), ("Costa Rica", "San José"), ("Panama", "Panama"),
    ("Paraguay", "Asunción"),
]

FORMULATIONS = [
    "Quelle est la capitale de {p} ?",
    "Donne la capitale du pays suivant : {p}.",
    "Capitale de {p} ?",
    "Quelle ville est la capitale de {p} ?",
    "Complete : la capitale de {p} est",
]


def normalise(s: str) -> str:
    """Lowercase, strip accents. Used for answer matching only."""
    s = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn").strip()


def fact_bank() -> list[dict]:
    """137 machine-checkable facts: 54 capitals, 24 elements, 24 additions."""
    facts = []
    for country, capital in CAPITALS:
        facts.append({"id": f"cap:{country}", "type": "capital",
                      "answers": [capital], "formulations": [f.format(p=country) for f in FORMULATIONS]})
    for name, symbol in ELEMENTS:
        facts.append({"id": f"elt:{name}", "type": "element", "answers": [symbol],
                      "formulations": [f"Quel est le symbole chimique du {name} ?",
                                       f"Donne le symbole chimique du {name}.",
                                       f"Symbole chimique du {name} ?",
                                       f"Quel symbole represente le {name} en chimie ?",
                                       f"Complete : le symbole chimique du {name} est"]})
    for a, b, s in ARITHMETIC:
        facts.append({"id": f"ari:{a}+{b}", "type": "arithmetic", "answers": [str(s)],
                      "formulations": [f"Combien font {a} + {b} ?", f"Calcule {a} + {b}.",
                                       f"{a} + {b} = ?", f"Quel est le resultat de {a} plus {b} ?",
                                       f"Complete : {a} + {b} ="]})
    return facts


# ---------------------------------------------------------------- model
def chat_ids(tokenizer, question: str):
    text = tokenizer.apply_chat_template([{"role": "user", "content": question}],
                                         add_generation_prompt=True, tokenize=False)
    return tokenizer(text, return_tensors="pt").input_ids


@torch.no_grad()
def generate(model, tokenizer, question: str, max_new_tokens: int = 24) -> str:
    ids = chat_ids(tokenizer, question)
    out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


@torch.no_grad()
def nll_answer(model, tokenizer, question: str, prefix: str, answer: str) -> float | None:
    """Mean NLL per answer token. Answer tokens isolated by character offsets,
    because the Qwen tokenizer merges a leading space with the following word."""
    base = tokenizer.apply_chat_template([{"role": "user", "content": question}],
                                         add_generation_prompt=True, tokenize=False)
    text = base + prefix + answer
    start = len(base + prefix)
    enc = tokenizer(text, return_offsets_mapping=True, return_tensors="pt")
    ids, offsets = enc["input_ids"], enc["offset_mapping"][0]
    idx = [i for i, (_, b) in enumerate(offsets.tolist()) if b > start and i > 0]
    if not idx:
        return None
    logp = model(input_ids=ids).logits[0].float().log_softmax(-1)
    return sum(float(-logp[i - 1, ids[0, i]]) for i in idx) / len(idx)


@torch.no_grad()
def capture(model, tokenizer, question: str, prefix: str = ""):
    """Residual stream at the last token, every layer. Shape (25, 896)."""
    base = tokenizer.apply_chat_template([{"role": "user", "content": question}],
                                         add_generation_prompt=True, tokenize=False)
    ids = tokenizer(base + prefix, return_tensors="pt").input_ids
    out = model(input_ids=ids, output_hidden_states=True)
    return torch.stack([h[0, -1] for h in out.hidden_states]).float().numpy()
