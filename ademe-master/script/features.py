"""
Simplify original dataset
from
"""

import pandas as pd

pd.options.display.max_columns = 100
pd.options.display.max_rows = 60
pd.options.display.max_colwidth = 100
pd.options.display.precision = 10
pd.options.display.width = 160
pd.set_option("display.float_format", "{:.4f}".format)
import numpy as np
import re
import typing as t
from collections import Counter
from sklearn.preprocessing import OrdinalEncoder
import json

if __name__ == "__main__":
    file = "./data/dpe-v2-tertiaire-2.csv"
    data = pd.read_csv(file)

    columns = data.columns.copy()
    columns = [col.lower() for col in columns]

    rgx = r"[°|/|']"
    columns = [re.sub(rgx, "_", col) for col in columns]

    rgx = r"²"
    columns = [re.sub(rgx, "2", col) for col in columns]
    rgx = r"[(|)]"
    columns = [re.sub(rgx, "", col) for col in columns]

    columns = [re.sub(r"é|è", "e", col) for col in columns]
    columns = [re.sub(r"_+", "_", col) for col in columns]

    data.columns = columns

    data.dropna(subset="etiquette_dpe", inplace=True)

    target = "etiquette_ges"
    target = "etiquette_dpe"
    id = "n_dpe"

    columns_categorical = [
        "version_dpe",
        "methode_du_dpe",
        "categorie_erp",
        "secteur_activite",
        "type_energie_principale_chauffage",
        "type_energie_n_1",
        "type_energie_n_2",
        "type_energie_n_3",
        "type_usage_energie_n_1",
        "type_usage_energie_n_2",
        "type_usage_energie_n_3",
    ]
    for col in columns_categorical:
        data[col].fillna("non renseigné", inplace=True)

    """
    encode type energies
    """
    type_energie = []
    for col in [
        "type_energie_principale_chauffage",
        "type_energie_n_1",
        "type_energie_n_2",
        "type_energie_n_3",
    ]:
        type_energie += list(data[col])

    type_energie_count = Counter(type_energie)
    print(type_energie_count.most_common(20))

    type_energie_map = {
        "non renseigné": "non renseigné",
        "Électricité": "Électricité",
        "Électricité d'origine renouvelable utilisée dans le bâtiment": "Électricité",
        "Gaz naturel": "Gaz naturel",
        "Butane": "GPL",
        "Propane": "GPL",
        "GPL": "GPL",
        "Fioul domestique": "Fioul domestique",
        "Réseau de Chauffage urbain": "Réseau de Chauffage urbain",
        "Charbon": "Combustible fossile",
        "autre combustible fossile": "Combustible fossile",
        "Bois – Bûches": "Bois",
        "Bois – Plaquettes forestières": "Bois",
        "Bois – Granulés (pellets) ou briquettes": "Bois",
        "Bois – Plaquettes d’industrie": "Bois",
    }

    for col in [
        "type_energie_principale_chauffage",
        "type_energie_n_1",
        "type_energie_n_2",
        "type_energie_n_3",
    ]:
        data[col] = data[col].apply(lambda d: type_energie_map[d])

    type_energie = []
    for col in [
        "type_energie_principale_chauffage",
        "type_energie_n_1",
        "type_energie_n_2",
        "type_energie_n_3",
    ]:
        type_energie += list(data[col])

    type_energie_count = Counter(type_energie)
    print(type_energie_count.most_common(20))

    """
    encode type usage
    """
    type_usage = []
    for col in [
        "type_usage_energie_n_1",
        "type_usage_energie_n_2",
        "type_usage_energie_n_3",
    ]:
        type_usage += list(data[col])

    type_usage_count = Counter(type_usage)
    print(type_usage_count.most_common(20))

    type_usage_map = {
        "non renseigné": "non renseigné",
        "périmètre de l'usage inconnu": "non renseigné",
        "Chauffage": "Chauffage",
        "Eau Chaude sanitaire": "Eau Chaude sanitaire",
        "Eclairage": "Eclairage",
        "Refroidissement": "Refroidissement",
        "Ascenseur(s)": "Ascenseur(s)",
        "auxiliaires et ventilation": "Refroidissement",
        "Autres usages": "Autres usages",
        "Bureautique": "Autres usages",
        "Abonnements": "Autres usages",
        "Production d'électricité à demeure": "Autres usages",
    }
    for col in [
        "type_usage_energie_n_1",
        "type_usage_energie_n_2",
        "type_usage_energie_n_3",
    ]:
        data[col] = data[col].apply(lambda d: type_usage_map[d])

    type_usage = []
    for col in [
        "type_usage_energie_n_1",
        "type_usage_energie_n_2",
        "type_usage_energie_n_3",
    ]:
        type_usage += list(data[col])

    type_usage_count = Counter(type_usage)
    print(type_usage_count.most_common(20))

    # encode categorical columns
    encoder = OrdinalEncoder()

    data[columns_categorical] = encoder.fit_transform(data[columns_categorical])
    for col in columns_categorical:
        data[col] = data[col].astype(int)

    mappings = {}
    for i, col in enumerate(encoder.feature_names_in_):
        mappings[col] = {
            int(value): category
            for value, category in enumerate(encoder.categories_[i])
        }

    # Save the mappings to a JSON file

    with open("./data/categorical_mappings.json", "w", encoding="utf-8") as f:
        json.dump(mappings, f, ensure_ascii=False, indent=4)

    columns_dates = ["date_visite_diagnostiqueur"]

    # supprimer le jour, garder annee et mois comme float
    col = "date_visite_diagnostiqueur"
    data[col] = data[col].apply(lambda d: float(".".join(d.split("-")[:2])))

    columns_float = [
        "conso_kwhep_m2_an",
        "emission_ges_kgco2_m2_an",
        "surface_utile",
        "conso_e_finale_energie_n_1",
        "conso_e_primaire_energie_n_1",
        "frais_annuel_energie_n_1",
        "conso_e_finale_energie_n_2",
        "conso_e_primaire_energie_n_2",
        "frais_annuel_energie_n_2",
        "conso_e_finale_energie_n_3",
        "conso_e_primaire_energie_n_3",
        "frais_annuel_energie_n_3",
    ]

    for col in columns_float:
        data[col].fillna(0.0, inplace=True)

    data["conso_finale_energie"] = 0.0
    data["conso_primaire_energie"] = 0.0
    data["frais_annuel_energie"] = 0.0
    for n in range(1, 4):
        col = f"conso_e_finale_energie_n_{n}"
        data["conso_finale_energie"] += data[col]
        col = f"conso_e_primaire_energie_n_{n}"
        data["conso_primaire_energie"] += data[col]
        col = f"frais_annuel_energie_n_{n}"
        data["frais_annuel_energie"] += data[col]

    columns_int = [
        "annee_construction",
        "nombre_occupant",
        "n_etage_appartement",
    ]

    for col in columns_int:
        data[col].fillna(-1, inplace=True)
        data[col] = data[col].astype(int)

    target_encoding = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
    data[target] = data[target].apply(lambda d: target_encoding[d])

    features = [
        "n_dpe",
        "version_dpe",
        "methode_du_dpe",
        "categorie_erp",
        "secteur_activite",
        "type_energie_principale_chauffage",
        "type_energie_n_1",
        "type_usage_energie_n_1",
        "conso_kwhep_m2_an",
        "emission_ges_kgco2_m2_an",
        "surface_utile",
        "conso_finale_energie",
        "conso_primaire_energie",
        "frais_annuel_energie",
        "annee_construction",
        target,
    ]

    data = data[features].copy()
    data.reset_index(inplace=True, drop=True)

    # alt

    data.to_csv("./data/dpe_tertiaire_20240307.csv", index=False)
