import json
import re
import os
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

# Download necessary NLTK data (run once)
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Path files
input_file = '/Users/duongcongthuyet/Downloads/workspace/AI /IT3180/data/locations.jsonl'
output_file = '/Users/duongcongthuyet/Downloads/workspace/AI /IT3180/data/locations_cleaned.jsonl'

# Check if input file exists
if not os.path.exists(input_file):
    print(f"Error: '{input_file}' not found.")
    print(f"Current working directory: {os.getcwd()}")
    exit(1)

# Read and clean data
cleaned_data = []
line_number = 0
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        line_number += 1
        stripped_line = line.strip()
        
        # Skip empty lines
        if not stripped_line:
            print(f"Warning: Line {line_number} is empty.")
            continue
        
        # JSON decode
        try:
            item = json.loads(stripped_line)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON on line {line_number}: {e}")
            continue
        
        name = item.get('location_name', '').strip()
        desc = item.get('location_description', '').strip()
        loc_type = item.get('location_type', '').strip()  # Added to use in classification
        
        # Skip if name or description is empty
        if not name or not desc:
            continue
        
        # Clean description: remove repetitive phrases and metadata
        desc = re.sub(r'This article is a stub.*?information!|Spoiler Warning!.*?(discretion|here)|"We acquire more knowledge.*?added\.|Behind the scenes.*?References.*$', 
                      '', desc, flags=re.DOTALL)
        desc = re.sub(r'\[.*?\]|\n|\s{2,}', ' ', desc).strip()
        
        cleaned_data.append({
            'location_name': name,
            'location_description': desc,
            'location_type': loc_type
        })

# Convert to DataFrame
df = pd.DataFrame(cleaned_data)

# Expanded keyword lists with lemmatized forms
action_keywords = ['battle', 'fight', 'experiment', 'confrontation', 'showdown', 'rescue', 'attack', 'duel', 'war', 'assault']
safe_keywords = ['safe', 'home', 'school', 'plan', 'refuge', 'hideout', 'base', 'shelter', 'protect', 'family', 'rest']
danger_keywords = ['monster', 'creature', 'beast', 'predator', 'hunt', 'kill', 'threat', 'dark', 'shadow', 'fog', 
                   'dimension', 'curse', 'vine', 'upside down', 'gate', 'portal', 'supernatural', 'haunted', 
                   'infest', 'demogorgon', 'mind flayer', 'vecna']

# Lemmatize keywords
action_keywords = [lemmatizer.lemmatize(kw) for kw in action_keywords]
safe_keywords = [lemmatizer.lemmatize(kw) for kw in safe_keywords]
danger_keywords = [lemmatizer.lemmatize(kw) for kw in danger_keywords]

# Labeling function
def classify_location(description, location_type=""):
    if not isinstance(description, str):
        return 'Minor Location'
    
    # Preprocess description: lowercase, strip, lemmatize
    desc = description.lower().strip()
    tokens = word_tokenize(desc)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Count keyword matches
    action_count = sum(1 for token in lemmatized_tokens if token in action_keywords)
    safe_count = sum(1 for token in lemmatized_tokens if token in safe_keywords)
    danger_count = sum(1 for token in lemmatized_tokens if token in danger_keywords)
    
    # Adjust scores based on location_type
    if location_type:
        type_lower = location_type.lower()
        if any(t in type_lower for t in ["school", "house", "residence", "home"]):
            safe_count += 1
        elif any(t in type_lower for t in ["military", "laboratory", "base", "facility", "prison"]):
            if danger_count > 0:
                danger_count += 1
            else:
                action_count += 1
    
    # If no keywords are found, return 'Minor Location'
    if action_count == 0 and safe_count == 0 and danger_count == 0:
        return 'Minor Location'
    
    # Determine the maximum count
    max_count = max(action_count, safe_count, danger_count)
    
    # Prioritize 'Action Hub' if action keywords are present
    if action_count > 0:
        return 'Action Hub'
    elif danger_count == max_count:
        return 'Danger Zone'
    elif safe_count == max_count:
        return 'Safe Haven'
    
    # Fallback (rare due to earlier checks)
    return 'Minor Location'

# Apply the classification
df['location_type'] = df.apply(lambda row: classify_location(row['location_description'], row['location_type']), axis=1)

# Save to file
with open(output_file, 'w', encoding='utf-8') as f:
    for _, row in df.iterrows():
        f.write(json.dumps(row.to_dict()) + '\n')

# Print result
print(f"Cleaned {len(df)} locations and saved to '{output_file}'.")
print(df)