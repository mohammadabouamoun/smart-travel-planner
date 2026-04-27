import pandas as pd
import numpy as np
import os

np.random.seed(42)

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

names = []
styles = []
for style, dests in destinations_by_style.items():
    for dest in dests:
        names.append(dest)
        styles.append(style)

extra_names = [
    "Lauterbrunnen, Switzerland", "Kotor, Montenegro", "Munnar, India",
    "Corfu, Greece", "Lake Bled, Slovenia", "Kazbegi, Georgia", "Moscow, Russia",
    "Warsaw, Poland", "Brasov, Romania", "Lviv, Ukraine", "Vilnius, Lithuania",
    "Tallinn, Estonia", "Helsinki, Finland", "Oslo, Norway", "Stockholm, Sweden",
    "Reykjavik, Iceland", "Dublin, Ireland", "Glasgow, Scotland", "Bristol, UK",
    "Lyon, France", "Marseille, France", "Nice, France", "Valencia, Spain",
    "Porto, Portugal", "Salzburg, Austria", "Innsbruck, Austria", "Brno, Czech Republic",
    "Leipzig, Germany", "Hamburg, Germany"
]
extra_styles = np.random.choice(['Adventure','Relaxation','Culture','Budget','Luxury','Family'],
                                 len(extra_names), p=[0.2,0.2,0.2,0.15,0.1,0.15])
names.extend(extra_names)
styles.extend(extra_styles)

n = len(names)

def get_feature_params(style):
    if style == 'Adventure':
        return {'temp_winter': (-10, 10), 'temp_summer': (10, 25), 'cost': (50, 150), 'beach_prob': 0.2, 'hiking_prob': 0.9, 'culture_prob': 0.4, 'nightlife_prob': 0.3, 'safety': (7,9)}
    elif style == 'Relaxation':
        return {'temp_winter': (20, 28), 'temp_summer': (25, 32), 'cost': (80, 200), 'beach_prob': 0.8, 'hiking_prob': 0.3, 'culture_prob': 0.4, 'nightlife_prob': 0.5, 'safety': (6,8)}
    elif style == 'Culture':
        return {'temp_winter': (0, 15), 'temp_summer': (20, 30), 'cost': (70, 180), 'beach_prob': 0.2, 'hiking_prob': 0.2, 'culture_prob': 0.9, 'nightlife_prob': 0.6, 'safety': (5,8)}
    elif style == 'Budget':
        return {'temp_winter': (15, 30), 'temp_summer': (25, 35), 'cost': (20, 70), 'beach_prob': 0.4, 'hiking_prob': 0.3, 'culture_prob': 0.6, 'nightlife_prob': 0.4, 'safety': (3,7)}
    elif style == 'Luxury':
        return {'temp_winter': (0, 20), 'temp_summer': (20, 30), 'cost': (200, 500), 'beach_prob': 0.5, 'hiking_prob': 0.2, 'culture_prob': 0.7, 'nightlife_prob': 0.8, 'safety': (8,10)}
    else:  # Family
        return {'temp_winter': (5, 20), 'temp_summer': (20, 28), 'cost': (80, 200), 'beach_prob': 0.4, 'hiking_prob': 0.3, 'culture_prob': 0.6, 'nightlife_prob': 0.2, 'safety': (8,10)}

data = []
for i in range(n):
    style = styles[i]
    params = get_feature_params(style)
    temp_winter = np.random.uniform(params['temp_winter'][0], params['temp_winter'][1])
    temp_summer = np.random.uniform(params['temp_summer'][0], params['temp_summer'][1])
    cost = np.random.randint(params['cost'][0], params['cost'][1])
    
    density_probs = None
    if style == 'Adventure':
        density_probs = {'low': 0.6, 'medium': 0.3, 'high': 0.1}
    elif style == 'Relaxation':
        density_probs = {'low': 0.3, 'medium': 0.5, 'high': 0.2}
    elif style == 'Culture':
        density_probs = {'low': 0.1, 'medium': 0.4, 'high': 0.5}
    elif style == 'Budget':
        density_probs = {'low': 0.1, 'medium': 0.3, 'high': 0.6}
    elif style == 'Luxury':
        density_probs = {'low': 0.2, 'medium': 0.6, 'high': 0.2}
    else:
        density_probs = {'low': 0.1, 'medium': 0.5, 'high': 0.4}
    density = np.random.choice(['low','medium','high'], p=[density_probs['low'], density_probs['medium'], density_probs['high']])
    
    beach = 1 if np.random.random() < params['beach_prob'] else 0
    hiking = 1 if np.random.random() < params['hiking_prob'] else 0
    culture = 1 if np.random.random() < params['culture_prob'] else 0
    nightlife = 1 if np.random.random() < params['nightlife_prob'] else 0
    safety = np.random.randint(params['safety'][0], params['safety'][1]+1)
    
    data.append({
        'name': names[i],
        'style': style,
        'avg_temp_winter': round(temp_winter, 1),
        'avg_temp_summer': round(temp_summer, 1),
        'cost_per_day': cost,
        'tourist_density': density,
        'has_beach': beach,
        'has_hiking': hiking,
        'has_culture': culture,
        'has_nightlife': nightlife,
        'safety_index': safety
    })

df = pd.DataFrame(data)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

os.makedirs('data', exist_ok=True)
df.to_csv('data/destinations.csv', index=False)
print(f"Saved {len(df)} destinations to data/destinations.csv")
print(df.head())
