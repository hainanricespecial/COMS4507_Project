import csv
import random
from pathlib import Path

# List of types from the Pokemon dataset
types = ["Grass", "Fire", "Water", "Normal", "Bug", "Poison", "Electric", "Ground", 
         "Rock", "Flying", "Psychic", "Dragon", "Ghost", "Dark", "Steel", "Fairy", 
         "Fighting", "Ice"]

# Read the CSV file
csv_path = "4POISONEDCSV/pokemon.csv"
rows = []
pokemon_names = []

with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)
        pokemon_names.append(row['Name'])

print(f"Total Pokemon: {len(pokemon_names)}")
print(f"Total possible types: {len(types)}")

# Randomly poison Type1, Type2, and Evolution columns
for row in rows:
    # Poison Type1 with 40% probability
    if random.random() < 0.4:
        row['Type1'] = random.choice(types)
    
    # Poison Type2 with 40% probability
    if random.random() < 0.4:
        row['Type2'] = random.choice(types) if random.random() < 0.5 else ""
    
    # Poison Evolution with 30% probability - use existing Pokemon names
    if random.random() < 0.3:
        row['Evolution'] = random.choice(pokemon_names) if random.random() < 0.8 else ""

# Write back to the CSV file
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['Name', 'Type1', 'Type2', 'Evolution'])
    writer.writeheader()
    writer.writerows(rows)

print(f"\nPoison complete! File saved to {csv_path}")
