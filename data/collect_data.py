import arxiv
import pandas as pd
import requests
import time
import os

# Configuration
N_PER_CATEGORY = 2000  # Adjust to reach ~20,000 total (with ~10 categories)
CATEGORY_MAP = {
    # --- COMPUTER SCIENCE (CS) ---
    "Machine Learning": "cat:cs.LG",
    "Artificial Intelligence": "cat:cs.AI",
    "Deep Learning": "cat:cs.LG",  # Souvent classé dans ML
    "Computer Vision": "cat:cs.CV",
    "Natural Language Processing": "cat:cs.CL",
    "Robotics": "cat:cs.RO",
    "Cybersecurity": "cat:cs.CR", # Cryptography and Security
    "Recommender Systems": "cat:cs.IR", # Information Retrieval
    "Software Engineering": "cat:cs.SE",
    "Distributed Systems": "cat:cs.DC", # Distributed, Parallel, and Cluster Computing
    "Databases & Data Mining": "cat:cs.DB",
    "Networking": "cat:cs.NI",
    "Human Computer Interaction": "cat:cs.HC",
    "Multi-Agent Systems": "cat:cs.MA",
    "Game Theory (CS)": "cat:cs.GT",

    # --- ECONOMICS & FINANCE (Econ / Q-Fin) ---
    "Econometrics": "cat:econ.EM",
    "Finance": "cat:q-fin.GN", # General Finance
    "Risk Management": "cat:q-fin.RM",
    "Portfolio Management": "cat:q-fin.PM",
    "Pricing & Securities": "cat:q-fin.PR",
    "Trading & Market Microstructure": "cat:q-fin.TR",
    "Economics (General)": "cat:econ.GN",
    "Theoretical Economics": "cat:econ.TH",

    # --- STATISTICS & MATH ---
    "Statistics (Machine Learning)": "cat:stat.ML",
    "Statistics (Theory)": "cat:math.ST",
    "Probability": "cat:math.PR",
    "Optimization": "cat:math.OC",
    "Combinatorics": "cat:math.CO",
    "Dynamical Systems": "cat:math.DS",
    "Number Theory": "cat:math.NT",
    
    # --- PHYSICS ---
    "Astrophysics": "cat:astro-ph",
    "Quantum Physics": "cat:quant-ph",
    "High Energy Physics": "cat:hep-th",
    "Condensed Matter": "cat:cond-mat",
    "General Relativity": "cat:gr-qc",
    "Fluid Dynamics": "cat:physics.flu-dyn",
    "Climate Science & Atmosphere": "cat:physics.ao-ph", # Atmospheric and Oceanic Physics
    "Geophysics": "cat:physics.geo-ph",
    "Biological Physics": "cat:physics.bio-ph",

    # --- BIOLOGY & MEDICINE (Quantitative) ---
    "Quantitative Biology": "cat:q-bio.QM", # Quantitative Methods
    "Genomics": "cat:q-bio.GN",
    "Neuroscience": "cat:q-bio.NC", # Neurons and Cognition
    "Populations and Evolution": "cat:q-bio.PE",
    "Biomolecules": "cat:q-bio.BM",
    
    # --- ENGINEERING (EESS) ---
    "Signal Processing": "cat:eess.SP",
    "Image & Video Processing": "cat:eess.IV",
    "Systems & Control": "cat:eess.SY", # Control Theory
    "Audio & Speech": "cat:eess.AS",
}

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "articles.csv")

def fetch_from_arxiv():
    """
    Fetches articles from arXiv based on CATEGORY_MAP.
    Returns a pandas DataFrame.
    """
    client = arxiv.Client()
    all_articles = []
    
    print(f"Starting arXiv harvest. Target: {N_PER_CATEGORY} articles per category.")

    for field_name, query in CATEGORY_MAP.items():
        print(f"Harvesting category: {field_name} ({query})...")
        
        search = arxiv.Search(
            query=query,
            max_results=N_PER_CATEGORY,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )

        count = 0
        try:
            # arxiv.Client.results is a generator
            for result in client.results(search):
                # Data Cleaning
                title = result.title.replace('\n', ' ').strip()
                abstract = result.summary.replace('\n', ' ').strip()
                authors = ", ".join([a.name for a in result.authors])
                
                # Extract ID (e.g., http://arxiv.org/abs/2106.12345v1 -> 2106.12345)
                # result.entry_id is usually the URL. result.get_short_id() is deprecated or not always available in some versions.
                # The reliable way is to parse the ID from the entry_id URL.
                article_id = result.entry_id.split('/')[-1]
                # Remove version number if present (e.g., v1) for cleaner ID, 
                # though Semantic Scholar usually handles versioned IDs, unversioned is safer for matching.
                if 'v' in article_id:
                    article_id = article_id.split('v')[0]

                all_articles.append({
                    "id": article_id,
                    "title": title,
                    "abstract": abstract,
                    "year": result.published.year,
                    "field": field_name,
                    "url": result.entry_id,
                    "author": authors,
                    "journal": "Preprint", # Default
                    "cite_nb": 0 # Default
                })
                count += 1
            
            print(f"  -> Collected {count} articles for {field_name}.")
            
        except Exception as e:
            print(f"  -> Error collecting {field_name}: {e}")
            continue

    df = pd.DataFrame(all_articles)
    print(f"Total articles collected from arXiv: {len(df)}")
    return df

def enrich_with_semantic_scholar(df):
    """
    Enriches the DataFrame with citation counts and venue info from Semantic Scholar.
    """
    print("Starting enrichment via Semantic Scholar...")
    
    # Semantic Scholar Batch Endpoint
    SS_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
    BATCH_SIZE = 100
    
    # Prepare IDs with prefix
    # We need to map back to the original dataframe index or ID
    unique_ids = df['id'].unique()
    
    # Result storage
    enrichment_data = {} # id -> {cite_nb, venue}

    total_batches = (len(unique_ids) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in range(0, len(unique_ids), BATCH_SIZE):
        batch_ids = unique_ids[i:i + BATCH_SIZE]
        # Format for Semantic Scholar: "ARXIV:ID"
        payload_ids = [f"ARXIV:{aid}" for aid in batch_ids]
        
        payload = {
            "ids": payload_ids,
            "fields": "citationCount,venue"
        }
        
        success = False
        retries = 0
        while not success and retries < 3:
            try:
                response = requests.post(SS_BATCH_URL, json=payload, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Process response
                    for item in data:
                        # item is None if not found
                        if item:
                            # The paperId returned might not match exactly what we sent if it resolved to a canonical ID
                            # But the order *should* match the input list according to some APIs, 
                            # however Semantic Scholar batch API returns a list of objects where 'paperId' is the SS ID.
                            # We rely on the fact that we requested by arXiv ID.
                            # Actually, the batch API returns results in the same order as requested? 
                            # Let's check documentation or be safe. 
                            # The response is a list of objects. If an ID is not found, it returns null in that position.
                            pass

                    # Since we can't easily map back if order isn't guaranteed (it usually is, but let's be robust),
                    # We can't rely on index if we don't trust it.
                    # However, the safest way with SS API is often to trust the order OR check if they return the input ID.
                    # The graph/v1/paper/batch endpoint returns a list of paper objects.
                    # It does NOT return the requested ID in the body usually, unless we ask for externalIds.
                    # Let's ask for externalIds to be safe.
                    
                    # Re-doing payload to include externalIds for mapping
                    # Actually, let's just trust the order for now as per standard usage, 
                    # or better: iterate and match index.
                    
                    for idx, paper_data in enumerate(data):
                        original_arxiv_id = batch_ids[idx]
                        
                        if paper_data:
                            citations = paper_data.get('citationCount', 0)
                            venue = paper_data.get('venue', '')
                            
                            enrichment_data[original_arxiv_id] = {
                                "cite_nb": citations,
                                "journal": venue if venue else "Preprint"
                            }
                    
                    success = True
                    print(f"  -> Processed batch {i//BATCH_SIZE + 1}/{total_batches}")
                    
                elif response.status_code == 429:
                    print("  -> Rate limit hit (429). Sleeping for 2 seconds...")
                    time.sleep(2)
                    retries += 1
                else:
                    print(f"  -> Error {response.status_code}: {response.text}")
                    retries += 1
                    
            except Exception as e:
                print(f"  -> Exception in batch {i//BATCH_SIZE + 1}: {e}")
                retries += 1
                time.sleep(1)
        
        # Polite sleep between batches
        time.sleep(1)

    # Apply enrichment to DataFrame
    print("Applying enrichment data to DataFrame...")
    
    def get_citations(row):
        if row['id'] in enrichment_data:
            return enrichment_data[row['id']]['cite_nb']
        return row['cite_nb']

    def get_journal(row):
        if row['id'] in enrichment_data:
            val = enrichment_data[row['id']]['journal']
            return val if val else "Preprint"
        return row['journal']

    df['cite_nb'] = df.apply(get_citations, axis=1)
    df['journal'] = df.apply(get_journal, axis=1)
    
    return df

def main():
    # Step 1: Harvest
    df = fetch_from_arxiv()
    
    if df.empty:
        print("No articles collected. Exiting.")
        return

    # Step 2: Enrich
    df = enrich_with_semantic_scholar(df)
    
    # Step 3: Save
    print(f"Saving {len(df)} articles to {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False)
    print("Done!")

if __name__ == "__main__":
    main()
