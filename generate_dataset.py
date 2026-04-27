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
# 3. Combine all names and styles
# ============================================================
names = []
styles = []
for style, dests in destinations_by_style.items():
    for dest in dests:
        names.append(dest)
        styles.append(style)

extra_names = list(extra_city_styles.keys())
for dest in extra_names:
    names.append(dest)
    styles.append(extra_city_styles[dest])

n = len(names)

# ============================================================
# 4. Improved feature generation helpers
# ============================================================
def random_normal_temp(low, high, mean=None, std=None):
    """
    Draw a temperature from a truncated normal distribution.
    If scipy not available, falls back to uniform.
    """
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
    'concentration' controls how tightly values cluster around the mean (higher = tighter).
    """
    mean = probability * 10
    # Adjust std so that ~95% of values are within [0,10]; smaller if mean near edges
    if mean <= 5:
        std = mean / 2.5   # ensures most values >0
    else:
        std = (10 - mean) / 2.5
    std = max(std, 0.5)   # avoid division by zero
    return np.clip(np.random.normal(mean, std/concentration), min_val, max_val)

def get_peak_season(winter_temp, summer_temp):
    """Determine peak season based on temperature contrast."""
    if winter_temp > summer_temp + 5:    # southern hemisphere (winter warmer than summer)
        return "winter"
    elif summer_temp > winter_temp + 10:
        return "summer"
    else:
        return "spring/fall"

def get_visa_prob(style):
    """Return (easy, medium, hard) probabilities for visa difficulty."""
    if style in ['Luxury', 'Family']:
        return [0.7, 0.25, 0.05]
    elif style == 'Culture':
        return [0.5, 0.35, 0.15]
    elif style in ['Adventure', 'Relaxation']:
        return [0.4, 0.4, 0.2]
    else: # Budget
        return [0.3, 0.4, 0.3]

def get_english_friendly_prob(style):
    """Probability that a destination is English‑friendly."""
    if style in ['Luxury', 'Family', 'Culture']:
        return 0.8
    elif style == 'Relaxation':
        return 0.7
    elif style == 'Adventure':
        return 0.6
    else: # Budget
        return 0.45

# Style parameter base definitions (used as defaults)
def get_feature_params(style):
    if style == 'Adventure':
        return {'temp_winter': (-10, 10), 'temp_summer': (10, 25), 'cost': (50, 150),
                'beach_prob': 0.2, 'hiking_prob': 0.9, 'culture_prob': 0.4, 'nightlife_prob': 0.3,
                'safety': (7, 9)}
    elif style == 'Relaxation':
        return {'temp_winter': (20, 28), 'temp_summer': (25, 32), 'cost': (80, 200),
                'beach_prob': 0.8, 'hiking_prob': 0.3, 'culture_prob': 0.4, 'nightlife_prob': 0.5,
                'safety': (6, 8)}
    elif style == 'Culture':
        return {'temp_winter': (0, 15), 'temp_summer': (20, 30), 'cost': (70, 180),
                'beach_prob': 0.2, 'hiking_prob': 0.2, 'culture_prob': 0.9, 'nightlife_prob': 0.6,
                'safety': (5, 8)}
    elif style == 'Budget':
        return {'temp_winter': (15, 30), 'temp_summer': (25, 35), 'cost': (20, 70),
                'beach_prob': 0.4, 'hiking_prob': 0.3, 'culture_prob': 0.6, 'nightlife_prob': 0.4,
                'safety': (3, 7)}
    elif style == 'Luxury':
        return {'temp_winter': (0, 20), 'temp_summer': (20, 30), 'cost': (200, 500),
                'beach_prob': 0.5, 'hiking_prob': 0.2, 'culture_prob': 0.7, 'nightlife_prob': 0.8,
                'safety': (8, 10)}
    else:  # Family
        return {'temp_winter': (5, 20), 'temp_summer': (20, 28), 'cost': (80, 200),
                'beach_prob': 0.4, 'hiking_prob': 0.3, 'culture_prob': 0.6, 'nightlife_prob': 0.2,
                'safety': (8, 10)}

# Density category to numeric mapping (and vice‑versa)
density_mapping = {
    'low': (5, 30),
    'medium': (30, 60),
    'high': (60, 95)
}

# ============================================================
# 5. Generate the dataset
# ============================================================
data = []
for i in range(n):
    style = styles[i]
    city_name = names[i]
    params = get_feature_params(style)

    # --- Temperatures (realistic truncated normal) ---
    temp_winter = round(random_normal_temp(params['temp_winter'][0], params['temp_winter'][1],
                                          mean=np.mean(params['temp_winter'])), 1)
    temp_summer = round(random_normal_temp(params['temp_summer'][0], params['temp_summer'][1],
                                          mean=np.mean(params['temp_summer'])), 1)

    # --- Cost (with possible luxury/beach correlation) ---
    base_cost = np.random.randint(params['cost'][0], params['cost'][1])
    beach = 1 if np.random.random() < params['beach_prob'] else 0
    # In luxury destinations, beach front often increases cost
    if style == 'Luxury' and beach:
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
    # Generate a continuous density score (0‑100) for the chosen category
    low_d, high_d = density_mapping[density_cat]
    density_score = np.random.uniform(low_d, high_d)

    # --- Continuous activity scores (0‑10) ---
    beach_score = random_score(params['beach_prob']) if beach else 0.0
    hiking_score = random_score(params['hiking_prob']) if np.random.random() < params['hiking_prob'] else 0.0
    culture_score = random_score(params['culture_prob']) if np.random.random() < params['culture_prob'] else 0.0
    nightlife_score = random_score(params['nightlife_prob']) if np.random.random() < params['nightlife_prob'] else 0.0

    # --- Safety (slight correlation with cost) ---
    base_safety_low, base_safety_high = params['safety']
    raw_safety = np.random.randint(base_safety_low, base_safety_high + 1)
    # Higher cost slightly boosts safety (up to +1.5 points, capped at 10)
    safety_bonus = (cost - np.mean(params['cost'])) / (np.ptp(params['cost']) + 1) * 1.5
    safety_index = min(10, max(1, round(raw_safety + safety_bonus)))

    # --- New features ---
    peak_season = get_peak_season(temp_winter, temp_summer)
    visa_probs = get_visa_prob(style)
    visa_difficulty = np.random.choice(['easy', 'medium', 'hard'], p=visa_probs)
    english_friendly = 1 if np.random.random() < get_english_friendly_prob(style) else 0

    # --- Assemble record ---
    data.append({
        'name': city_name,
        'style': style,
        'avg_temp_winter': temp_winter,
        'avg_temp_summer': temp_summer,
        'cost_per_day': cost,
        'tourist_density': density_cat,            # categorical for filtering
        'density_score': round(density_score, 1),  # numeric for regression
        'has_beach': beach,                        # keep original binary for compatibility
        'beach_score': round(beach_score, 1),
        'hiking_score': round(hiking_score, 1),
        'culture_score': round(culture_score, 1),
        'nightlife_score': round(nightlife_score, 1),
        'safety_index': safety_index,
        'peak_season': peak_season,
        'visa_difficulty': visa_difficulty,
        'english_friendly': english_friendly
    })

# ============================================================
# 6. Create DataFrame and shuffle
# ============================================================
df = pd.DataFrame(data)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# ============================================================
# 7. Save data and metadata
# ============================================================
os.makedirs('data', exist_ok=True)

# Use timestamp to avoid overwriting (unless you explicitly want to)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = f'data/destinations_{timestamp}.csv'
df.to_csv(csv_path, index=False)

# Also save the most recent version as 'destinations.csv' for convenience
df.to_csv('data/destinations.csv', index=False)

# Metadata JSON
metadata = {
    "description": "Synthetic travel destination dataset",
    "num_samples": len(df),
    "features": {
        "name": "Destination name (city, country)",
        "style": "Primary travel style (one of: Adventure, Relaxation, Culture, Budget, Luxury, Family)",
        "avg_temp_winter": "Average winter temperature (°C)",
        "avg_temp_summer": "Average summer temperature (°C)",
        "cost_per_day": "Estimated daily cost (USD, accommodation + food + activities)",
        "tourist_density": "Subjective tourism density category (low/medium/high)",
        "density_score": "Continuous 0-100 score of tourist density (higher = more crowded)",
        "has_beach": "Binary: does the destination have beaches? (1=yes)",
        "beach_score": "Beach quality/availability score (0-10)",
        "hiking_score": "Hiking and outdoor adventure score (0-10)",
        "culture_score": "Cultural attractions score (0-10)",
        "nightlife_score": "Nightlife score (0-10)",
        "safety_index": "Safety perception index (1=very unsafe, 10=extremely safe)",
        "peak_season": "Most popular tourist season ('winter', 'summer', or 'spring/fall')",
        "visa_difficulty": "Ease of obtaining a tourist visa ('easy', 'medium', 'hard')",
        "english_friendly": "Binary: is English widely spoken? (1=yes)"
    },
    "generation_notes": "Temperature sampled via truncated normal (or uniform fallback). Scores correlate with style probabilities. Extra cities have manually curated styles for realism.",
    "style_distribution": df['style'].value_counts().to_dict(),
    "created_at": timestamp
}
with open(f'data/metadata_{timestamp}.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"Dataset saved with {len(df)} destinations.")
print(f"Main file: data/destinations.csv")
print(f"Versioned file: {csv_path}")
print(df.head())