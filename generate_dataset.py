import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

# Optional: use scipy for more realistic temperature distributions
try:
    from scipy.stats import truncnorm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("scipy not installed, falling back to uniform temperatures. Install scipy for better distributions.")

np.random.seed(42)

# ============================================================
# 1. Base destination definitions (main styles)
# ============================================================
destinations_by_style = {
    'Adventure': [
        "Interlaken, Switzerland", "Queenstown, New Zealand", "Banff, Canada",
        "Chamonix, France", "Zhangjiajie, China", "Cusco, Peru", "Moab, USA",
        "Torres del Paine, Chile", "Manali, India", "Nepal (Kathmandu)",
        "Costa Rica (La Fortuna)", "Patagonia, Argentina", "Swiss Alps",
        "New Zealand South Island", "Norwegian Fjords", "Alps, Austria",
        "Himalayas, Nepal", "Rocky Mountains, USA", "Cappadocia, Turkey",
        "Krabi, Thailand", "Santorini, Greece", "Madeira, Portugal",
        "Transylvania, Romania", "Sapa, Vietnam", "Mount Everest Base Camp"
    ],
    'Relaxation': [
        "Maldives", "Bali, Indonesia", "Phuket, Thailand", "Seychelles",
        "Maui, Hawaii", "Fiji", "Bahamas", "Cancun, Mexico", "Sardinia, Italy",
        "Algarve, Portugal", "Mauritius", "Krabi, Thailand", "Bora Bora",
        "Costa del Sol, Spain", "Goa, India", "Phu Quoc, Vietnam",
        "Zanzibar, Tanzania", "Palawan, Philippines", "Barbados", "Malta",
        "Croatian Coast (Dubrovnik)", "Amalfi Coast, Italy", "Riviera Maya, Mexico"
    ],
    'Culture': [
        "Rome, Italy", "Paris, France", "Kyoto, Japan", "Istanbul, Turkey",
        "Cairo, Egypt", "Beijing, China", "Athens, Greece", "Jerusalem, Israel",
        "Vienna, Austria", "Prague, Czech Republic", "Florence, Italy",
        "St. Petersburg, Russia", "Mexico City, Mexico", "Lisbon, Portugal",
        "Jaipur, India", "Marrakech, Morocco", "Tehran, Iran", "Hanoi, Vietnam",
        "Bangkok, Thailand", "Edinburgh, Scotland", "Budapest, Hungary",
        "Krakow, Poland", "Seville, Spain", "Tbilisi, Georgia"
    ],
    'Budget': [
        "Hanoi, Vietnam", "Bangkok, Thailand", "Delhi, India", "Cairo, Egypt",
        "Medellin, Colombia", "Bogota, Colombia", "Lima, Peru", "Kathmandu, Nepal",
        "Phnom Penh, Cambodia", "Vientiane, Laos", "Belgrade, Serbia",
        "Sofia, Bulgaria", "Kiev, Ukraine", "Tbilisi, Georgia", "Algiers, Algeria",
        "Karachi, Pakistan", "Dhaka, Bangladesh", "La Paz, Bolivia",
        "Managua, Nicaragua", "San Salvador, El Salvador", "Minsk, Belarus"
    ],
    'Luxury': [
        "Monaco", "St. Moritz, Switzerland", "Aspen, USA", "Dubai, UAE",
        "Saint-Tropez, France", "Hamptons, USA", "Cannes, France",
        "Lake Como, Italy", "Beverly Hills, USA", "Abu Dhabi, UAE",
        "Singapore", "Hong Kong", "Tokyo, Japan", "Gstaad, Switzerland",
        "Courchevel, France", "Marbella, Spain", "Portofino, Italy",
        "Bahrain", "Doha, Qatar", "The Maldives"
    ],
    'Family': [
        "Orlando, USA", "Anaheim, USA", "Tokyo, Japan", "Paris, France",
        "Gold Coast, Australia", "San Diego, USA", "Copenhagen, Denmark",
        "Rotterdam, Netherlands", "Singapore", "London, UK",
        "Chicago, USA", "Barcelona, Spain", "Rome, Italy", "Sydney, Australia",
        "Vancouver, Canada", "Washington DC, USA"
    ]
}

# ============================================================
# 2. Extra cities with manually assigned realistic styles
#    (prevents Moscow from becoming a beach destination)
# ============================================================
extra_city_styles = {
    "Lauterbrunnen, Switzerland": "Adventure",
    "Kotor, Montenegro": "Relaxation",
    "Munnar, India": "Relaxation",
    "Corfu, Greece": "Relaxation",
    "Lake Bled, Slovenia": "Adventure",
    "Kazbegi, Georgia": "Adventure",
    "Moscow, Russia": "Culture",
    "Warsaw, Poland": "Culture",
    "Brasov, Romania": "Adventure",
    "Lviv, Ukraine": "Culture",
    "Vilnius, Lithuania": "Culture",
    "Tallinn, Estonia": "Culture",
    "Helsinki, Finland": "Culture",
    "Oslo, Norway": "Adventure",
    "Stockholm, Sweden": "Culture",
    "Reykjavik, Iceland": "Adventure",
    "Dublin, Ireland": "Culture",
    "Glasgow, Scotland": "Culture",
    "Bristol, UK": "Culture",
    "Lyon, France": "Culture",
    "Marseille, France": "Culture",
    "Nice, France": "Luxury",
    "Valencia, Spain": "Culture",
    "Porto, Portugal": "Culture",
    "Salzburg, Austria": "Culture",
    "Innsbruck, Austria": "Adventure",
    "Brno, Czech Republic": "Culture",
    "Leipzig, Germany": "Culture",
    "Hamburg, Germany": "Culture"
}

# ============================================================
# 3. Combine all names and styles (deduplicated, manual overrides)
# ============================================================
style_map = {}
# First, load base listings (later entries overwrite earlier ones)
for style, dests in destinations_by_style.items():
    for dest in dests:
        style_map[dest] = style
# Then apply manual overrides – these have the final say
style_map.update(extra_city_styles)

names = list(style_map.keys())
styles = [style_map[name] for name in names]
n = len(names)

# ============================================================
# 4. Helper: identify Southern Hemisphere destinations
# ============================================================
SOUTHERN_DESTINATIONS = {
    "Queenstown, New Zealand", "Cusco, Peru", "Torres del Paine, Chile",
    "Patagonia, Argentina", "New Zealand South Island", "Bora Bora",
    "Mauritius", "Seychelles", "Fiji", "Zanzibar, Tanzania",
    "Bali, Indonesia", "La Paz, Bolivia", "Gold Coast, Australia",
    "Sydney, Australia", "Lima, Peru", "Phnom Penh, Cambodia",  # actually Phnom Penh is northern,
    # but we'll keep the list simple – real‑world usage may require manual check
}

def get_hemisphere(dest_name):
    return 'South' if dest_name in SOUTHERN_DESTINATIONS else 'North'

# ============================================================
# 5. Improved feature generation helpers
# ============================================================
def random_normal_temp(low, high, mean=None, std=None):
    """Draw a temperature from a truncated normal distribution (or uniform fallback)."""
    if SCIPY_AVAILABLE:
        if mean is None:
            mean = (low + high) / 2
        if std is None:
            std = (high - low) / 4   # ~95% of values inside [low, high]
        a = (low - mean) / std
        b = (high - mean) / std
        return truncnorm.rvs(a, b, loc=mean, scale=std)
    else:
        return np.random.uniform(low, high)

def random_score(probability, min_val=0, max_val=10, concentration=5):
    """
    Generate a continuous score (0-10) based on a style's activity probability.
    Uses a normal distribution centered at probability * 10, clipped to [0,10].
    """
    mean = probability * 10
    # Adjust std so that ~95% of values are within [0,10]
    if mean <= 5:
        std = mean / 2.5
    else:
        std = (10 - mean) / 2.5
    std = max(std, 0.5)
    return np.clip(np.random.normal(mean, std/concentration), min_val, max_val)

def get_peak_season(temp_diff):
    """Determine peak season based on temperature contrast between warmest and coldest month."""
    if temp_diff > 10:
        return "summer"
    else:
        return "spring/fall"

def get_visa_prob(style):
    """Return probabilities for visa difficulty (easy, medium, hard)."""
    if style in ['Luxury', 'Family']:
        return [0.7, 0.25, 0.05]
    elif style == 'Culture':
        return [0.5, 0.35, 0.15]
    elif style in ['Adventure', 'Relaxation']:
        return [0.4, 0.4, 0.2]
    else:  # Budget
        return [0.3, 0.4, 0.3]

def get_english_friendly_prob(style):
    """Probability that a destination is English‑friendly."""
    if style in ['Luxury', 'Family', 'Culture']:
        return 0.8
    elif style == 'Relaxation':
        return 0.7
    elif style == 'Adventure':
        return 0.6
    else:  # Budget
        return 0.45

# Style parameters – now provide ranges for coldest and warmest months
def get_feature_params(style):
    if style == 'Adventure':
        return {'temp_cold': (-10, 10), 'temp_warm': (10, 25), 'cost': (50, 150),
                'beach_prob': 0.2, 'hiking_prob': 0.9, 'culture_prob': 0.4, 'nightlife_prob': 0.3,
                'safety': (7, 9)}
    elif style == 'Relaxation':
        return {'temp_cold': (20, 28), 'temp_warm': (25, 32), 'cost': (80, 200),
                'beach_prob': 0.8, 'hiking_prob': 0.3, 'culture_prob': 0.4, 'nightlife_prob': 0.5,
                'safety': (6, 8)}
    elif style == 'Culture':
        return {'temp_cold': (0, 15), 'temp_warm': (20, 30), 'cost': (70, 180),
                'beach_prob': 0.2, 'hiking_prob': 0.2, 'culture_prob': 0.9, 'nightlife_prob': 0.6,
                'safety': (5, 8)}
    elif style == 'Budget':
        return {'temp_cold': (15, 30), 'temp_warm': (25, 35), 'cost': (20, 70),
                'beach_prob': 0.4, 'hiking_prob': 0.3, 'culture_prob': 0.6, 'nightlife_prob': 0.4,
                'safety': (3, 7)}
    elif style == 'Luxury':
        return {'temp_cold': (0, 20), 'temp_warm': (20, 30), 'cost': (200, 500),
                'beach_prob': 0.5, 'hiking_prob': 0.2, 'culture_prob': 0.7, 'nightlife_prob': 0.8,
                'safety': (8, 10)}
    else:  # Family
        return {'temp_cold': (5, 20), 'temp_warm': (20, 28), 'cost': (80, 200),
                'beach_prob': 0.4, 'hiking_prob': 0.3, 'culture_prob': 0.6, 'nightlife_prob': 0.2,
                'safety': (8, 10)}

# Density category to numeric mapping
density_mapping = {
    'low': (5, 30),
    'medium': (30, 60),
    'high': (60, 95)
}

# ============================================================
# 6. Generate the dataset
# ============================================================
data = []
for i in range(n):
    style = styles[i]
    city_name = names[i]
    params = get_feature_params(style)

    # --- Temperatures (realistic truncated normal) ---
    temp_coldest = round(random_normal_temp(params['temp_cold'][0], params['temp_cold'][1],
                                           mean=np.mean(params['temp_cold'])), 1)
    temp_warmest = round(random_normal_temp(params['temp_warm'][0], params['temp_warm'][1],
                                           mean=np.mean(params['temp_warm'])), 1)

    # --- Cost (with possible luxury/beach correlation) ---
    base_cost = np.random.randint(params['cost'][0], params['cost'][1])

    # --- Activity scores (always generated, no zero‑cut gate) ---
    beach_score = round(random_score(params['beach_prob']), 1)
    has_beach = 1 if beach_score >= 1.0 else 0   # derive binary from continuous score (threshold 1)
    hiking_score = round(random_score(params['hiking_prob']), 1)
    culture_score = round(random_score(params['culture_prob']), 1)
    nightlife_score = round(random_score(params['nightlife_prob']), 1)

    # Cost boost for luxury beach destinations (correlation)
    if style == 'Luxury' and has_beach:
        cost = int(base_cost * np.random.uniform(1.1, 1.3))
    else:
        cost = base_cost

    # --- Tourist density (categorical + numeric) ---
    density_probs = {
        'Adventure': [0.6, 0.3, 0.1],
        'Relaxation': [0.3, 0.5, 0.2],
        'Culture': [0.1, 0.4, 0.5],
        'Budget': [0.1, 0.3, 0.6],
        'Luxury': [0.2, 0.6, 0.2],
        'Family': [0.1, 0.5, 0.4]
    }
    density_cat = np.random.choice(['low', 'medium', 'high'],
                                   p=density_probs.get(style, [0.2, 0.5, 0.3]))
    low_d, high_d = density_mapping[density_cat]
    density_score = round(np.random.uniform(low_d, high_d), 1)

    # --- Safety (slight correlation with cost) ---
    base_safety_low, base_safety_high = params['safety']
    raw_safety = np.random.randint(base_safety_low, base_safety_high + 1)
    safety_bonus = (cost - np.mean(params['cost'])) / (np.ptp(params['cost']) + 1) * 1.5
    safety_index = min(10, max(1, round(raw_safety + safety_bonus)))

    # --- New features ---
    hemisphere = get_hemisphere(city_name)
    peak_season = get_peak_season(temp_warmest - temp_coldest)
    visa_probs = get_visa_prob(style)
    visa_difficulty = np.random.choice(['easy', 'medium', 'hard'], p=visa_probs)
    english_friendly = 1 if np.random.random() < get_english_friendly_prob(style) else 0

    # --- Assemble record ---
    data.append({
        'name': city_name,
        'style': style,
        'temp_coldest_month': temp_coldest,
        'temp_warmest_month': temp_warmest,
        'cost_per_day': cost,
        'tourist_density': density_cat,
        'density_score': density_score,
        'has_beach': has_beach,
        'beach_score': beach_score,
        'hiking_score': hiking_score,
        'culture_score': culture_score,
        'nightlife_score': nightlife_score,
        'safety_index': safety_index,
        'hemisphere': hemisphere,
        'peak_season': peak_season,
        'visa_difficulty': visa_difficulty,
        'english_friendly': english_friendly
    })

# ============================================================
# 7. Create DataFrame and shuffle
# ============================================================
df = pd.DataFrame(data)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# ============================================================
# 8. Save data and metadata
# ============================================================
os.makedirs('data', exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = f'data/destinations_{timestamp}.csv'
df.to_csv(csv_path, index=False)
df.to_csv('data/destinations.csv', index=False)  # convenience copy

metadata = {
    "description": "Synthetic travel destination dataset (improved v2)",
    "num_samples": len(df),
    "features": {
        "name": "Destination name (city, country)",
        "style": "Primary travel style (Adventure, Relaxation, Culture, Budget, Luxury, Family)",
        "temp_coldest_month": "Average temperature of the coldest month (°C)",
        "temp_warmest_month": "Average temperature of the warmest month (°C)",
        "cost_per_day": "Estimated daily cost (USD, accommodation + food + activities)",
        "tourist_density": "Subjective tourism density category (low/medium/high)",
        "density_score": "Continuous 0‑100 score of tourist density (higher = more crowded)",
        "has_beach": "Binary: does the destination have good beaches? (1=yes; derived from beach_score ≥ 1)",
        "beach_score": "Beach quality/availability score (0‑10)",
        "hiking_score": "Hiking and outdoor adventure score (0‑10)",
        "culture_score": "Cultural attractions score (0‑10)",
        "nightlife_score": "Nightlife score (0‑10)",
        "safety_index": "Safety perception index (1=very unsafe, 10=extremely safe)",
        "hemisphere": "Hemisphere (North/South) – based on curated list",
        "peak_season": "Most popular tourist season ('summer' or 'spring/fall') – derived from temperature contrast",
        "visa_difficulty": "Ease of obtaining a tourist visa ('easy', 'medium', 'hard')",
        "english_friendly": "Binary: is English widely spoken? (1=yes)"
    },
    "generation_notes": (
        "v2 improvements: deduplicated city names, manual style overrides applied, "
        "temperature columns renamed to coldest/warmest month (hemisphere‑agnostic), "
        "activity scores are always continuous (no zero‑cut gate), beach_score now consistent with has_beach."
    ),
    "style_distribution": df['style'].value_counts().to_dict(),
    "created_at": timestamp
}
with open(f'data/metadata_{timestamp}.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"Dataset saved with {len(df)} destinations.")
print(f"Main file: data/destinations.csv")
print(f"Versioned file: {csv_path}")
print(df.head())