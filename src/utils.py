import requests
import time
import math
from datetime import datetime

# email requested by OpenAlex
MY_EMAIL = "aurelvvince@gmail.com"

def reconstruct_abstract(inverted_index):
    """
    Reconstruit le texte lisible d'un abstract à partir de l'index inversé d'OpenAlex.

    OpenAlex ne stock pas le texte brut de l'abstratc pour des raisons de droits d'auteur (contournement 
    des restrictions des éditeurs) et de compression mais fournit un dict où chaque 
    mot est associé à sa position dans la phrase.
    par ex :
        Entrée : {"L'IA": [0], "futur": [3], "le": [2], "est": [1]}
        Sortie : "L'IA est le futur"

    Args:
        inverted_index (dict): Dictionnaire {mot: [position]}.

    Returns:
        str: Le texte reconstruit dans l'ordre chronologique.
    """
    if not inverted_index:
        return ""
    word_list = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_list.append((pos, word))
    sorted_words = sorted(word_list, key=lambda x: x[0])
    return " ".join([w[1] for w in sorted_words])


def get_concept_id(concept_name):
    """Récupère l'ID OpenAlex d'un concept (ex: Machine Learning -> C154945302)"""
    url = "https://api.openalex.org/concepts"
    params = {"search": concept_name, "per_page": 1}
    headers = {"User-Agent": f"mailto:{MY_EMAIL}"}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=10)
        if r.status_code == 200:
            results = r.json().get('results', [])
            if results:
                return results[0]['id']
    except Exception as e:
        print(f"Error finding concept '{concept_name}': {e}")
    return None


# ---------- IMAGE MAPPING ----------

FIELD_TO_IMAGE_MAP = {
    "computer_science_and_ai.png": [
        "machine_learning", "deep_learning", "artificial_intelligence", "computer_vision", "nlp", "robotics", 
        "neural_network", "gan", "transformer", "llm", "recommender_system", "data_mining", "big_data", "hci", 
        "cybersecurity", "blockchain", "cloud_computing", "iot", "software_engineering", "database"
    ],
    "economics_finance_and_business.png": [
        "econometrics", "macroeconomics", "microeconomics", "finance", "asset_pricing", "portfolio_optimization", 
        "risk_management", "corporate_finance", "fintech", "crypto", "marketing", "supply_chain", "game_theory"
    ],
    "mathematics_and_statistics.png": [
        "statistics", "probability", "bayesian_inference", "time_series", "optimization", "linear_algebra", 
        "topology", "differential_equation", "combinatorics", "number_theory", "graph_theory"
    ],
    "physics_and_astronomy.png": [
        "physics", "quantum_mechanics", "astrophysics", "cosmology", "relativity", "nanotechnology", "optics", 
        "fluid_dynamics", "nuclear_physics"
    ],
    "chemistry_and_materials.png": [
        "chemistry", "organic_chemistry", "inorganic_chemistry", "biochemistry", "polymer", "nanomaterials", 
        "chemical_engineering", "crystallography"
    ],
    "biology_and_medicine.png": [
        "biology", "genetics", "genomics", "neuroscience", "cognitive_science", "immunology", "virology", 
        "epidemiology", "public_health", "medicine", "pharmacology", "crispr"
    ],
    "earth_and_environment.png": [
        "environmental_science", "climate_change", "ecology", "biodiversity", "oceanography", "geology", 
        "renewable_energy", "solar_power", "wind_power", "sustainability", "agriculture"
    ],
    "engineering.png": [
        "engineering", "electrical", "mechanical", "civil", "aerospace", "control_theory", "signal_processing", 
        "telecommunications", "electronics", "3d_printing"
    ],
    "social_sciences.png": [
        "psychology", "sociology", "political_science", "international_relations", "education", "law", 
        "philosophy", "urban_planning", "smart_city"
    ]
}

# Invert the map for O(1) lookup: { "machine_learning": "computer_science_and_ai.png", ... }
FIELD_LOOKUP = {}
for img, fields in FIELD_TO_IMAGE_MAP.items():
    for f in fields:
        # Normalize keys: lowercase and replace spaces with underscores just in case
        key = f.lower().replace(" ", "_")
        FIELD_LOOKUP[key] = img

def get_article_image(field_name):
    """
    Returns the static image path based on the article's field.
    """
    if not field_name:
        return "/static/img/computer_science_and_ai.png" # Default fallback
    
    # Normalize input field
    normalized_field = str(field_name).lower().strip().replace(" ", "_")
    
    # Lookup
    image_filename = FIELD_LOOKUP.get(normalized_field)
    
    if image_filename:
        return f"/static/img/{image_filename}"
    
    # Fallback if field not found
    return "/static/img/computer_science_and_ai.png"


def fetch_papers_by_concept(concept_id, concept_name, total_limit):
    """
    Récupère les articles de manière représentative (Uniforme sur les années).
    Stratégie : Loop 2010 -> 2025 et prend ~X articles par an.
    """
    works_url = "https://api.openalex.org/works" # url
    collected = []
    
    # Définition de la plage temporelle de 2010 a ajd
    start_year = 2010
    current_year = datetime.now().year + 1 # Inclure 2025/2026
    years = list(range(start_year, current_year))
    
    # Quota de papiers a prendre par année (750 articles / 16 ans (2010-2025) = 46 articles/an)
    limit_per_year = math.ceil(total_limit / len(years))

    select_fields = (
        "id,title,abstract_inverted_index,publication_year,"
        "primary_location,authorships,cited_by_count,doi,ids"
    )

    print(f"   -> Harvesting '{concept_name}' (~{limit_per_year} papers/year)...")

    for year in years:
        # Bloqué si limit atteinte (pour sécu la collect)
        if len(collected) >= total_limit:
            break

        params = {
            "filter": f"concepts.id:{concept_id},publication_year:{year}",
            "select": select_fields,
            "per_page": limit_per_year, # On prend exactement le quota requis sur l'année
            # most cited par an
            "sort": "cited_by_count:desc" 
        }
        headers = {"User-Agent": f"mailto:{MY_EMAIL}"}

        try:
            # requete
            r = requests.get(works_url, params=params, headers=headers, timeout=10)
            
            if r.status_code != 200:
                print(f"      Error year {year}: {r.status_code}") # msg d'erreur si ca fail
                continue
            
            data = r.json()
            results = data.get('results', [])
            
            if not results:
                continue

            for work in results:
                # Vérification qualité data (Titre + Abstract obligatoires)
                if not work.get('title') or not work.get('abstract_inverted_index'):
                    continue
                
                # Extract auteurs names
                authors = []
                for auth in work.get('authorships', []):
                    if auth.get('author'):
                        authors.append(auth['author'].get('display_name', ''))
                author_str = ", ".join(authors)

                # Extract Journal
                journal = "Unknown"
                loc = work.get('primary_location') or {}
                if loc.get('source'):
                    journal = loc['source'].get('display_name', 'Unknown')

                # Extract URL
                url = work.get('doi') or work.get('ids', {}).get('openalex', '')

                # dcp stocker tout dans la list
                collected.append({
                    "id": work['id'].replace("https://openalex.org/", ""),
                    "title": work['title'],
                    "abstract": reconstruct_abstract(work['abstract_inverted_index']),
                    "year": work['publication_year'],
                    "field": concept_name,
                    "url": url,
                    "author": author_str,
                    "journal": journal,
                    "cite_nb": work['cited_by_count']
                })

            # Petite pause pour l'API (eviter de se faire eject)
            time.sleep(0.1)

        except Exception as e:
            print(f"      Crash year {year}: {e}")
            continue
            
    return collected