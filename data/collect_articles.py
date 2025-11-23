import pandas as pd
import os
from src.utils import (
    get_concept_id, fetch_papers_by_concept, 
)
from src.config import N_PER_CATEGORY, OUTPUT_FILE, CATEGORY_LIST

# MAIN

def main():
    all_data = []
    
    # Création dossier si no
    output_dir = os.path.dirname(OUTPUT_FILE)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except OSError:
            pass 

    print(f"Starting OpenAlex collection (2010-2025). Target: {N_PER_CATEGORY} per category.")
      
    for category in CATEGORY_LIST:
        # Pour chaque category:
        # 1. On prend l'ID du concept
        cid = get_concept_id(category)
        
        if not cid:
            print(f"Concept ID not found for '{category}': Pass") # si not found on passe
            continue
            
        # 2. On recup les articles
        works = fetch_papers_by_concept(cid, category, total_limit=N_PER_CATEGORY)
        print(f"   -> Collected {len(works)} articles.")
        
        all_data.extend(works)
        
        # Sauvegarde (tous les 1000 articlesn, on save)
        if len(all_data) % 1000 < len(works):
             print("   (Saving intermediate backup...)")
             pd.DataFrame(all_data).to_csv(OUTPUT_FILE, index=False)

    # Final Save
    if all_data:
        df = pd.DataFrame(all_data)
        # Suppression doublons (car un article peut être taggé ML et AI en mm temps)
        df = df.drop_duplicates(subset=['id'])
        
        print(f"\nDONE! Total articles: {len(df)}")
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"Saved to {OUTPUT_FILE}")
    else:
        print("No data collected.")

if __name__ == "__main__":
    main()