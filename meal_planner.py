"""
AI Child Meal Planner & Pediatric Nutrition Engine (1–5 Years)
References & Standards:
- AskNestle Meal Plan (https://www.asknestle.in/meal-plan)
- ICMR - National Institute of Nutrition (NIN) RDA 2024 & Dietary Guidelines for Indians
- NIOS Meal Planning (Lesson 5)
- USDA MyPlate / FDA Nutrition Facts Guidelines
- Harvard Healthy Eating Plate & WHO Healthy Diet Guidelines
- Precision Nutrition Hand-Size Portion Guide
"""

from io import BytesIO
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas

# --- ICMR-NIN RDA 2024 & AskNestle Nutritional Baselines ---
RDA_TABLE = {
    "1-3_years": {
        "calories": 1010,
        "protein_g": 12.5,
        "fat_g": 27.0,
        "carbs_g": 130.0,
        "calcium_mg": 500,
        "iron_mg": 8.0,
        "zinc_mg": 5.0,
        "vit_a_mcg": 390,
        "fiber_g": 15.0,
        "water_ml": 1200
    },
    "3-5_years": {
        "calories": 1360,
        "protein_g": 16.0,
        "fat_g": 35.0,
        "carbs_g": 170.0,
        "calcium_mg": 550,
        "iron_mg": 11.0,
        "zinc_mg": 6.0,
        "vit_a_mcg": 400,
        "fiber_g": 18.0,
        "water_ml": 1500
    }
}

# --- Hand-Size Portion Guidelines ---
HAND_PORTION_GUIDE = {
    "Palm": {
        "nutrient": "Protein (Dal, Paneer, Egg, Fish/Chicken, Tofu)",
        "serving_1_3": "1/2 child's palm (~15-20g cooked) per meal",
        "serving_3_5": "1 child's palm (~25-35g cooked) per meal",
        "icon": "✋"
    },
    "Fist": {
        "nutrient": "Vegetables & Seasonal Fruits",
        "serving_1_3": "1 child's fist (~50g) at 2-3 meals/day",
        "serving_3_5": "1 child's fist (~75-100g) at 3-4 meals/day",
        "icon": "✊"
    },
    "Cupped Hand": {
        "nutrient": "Grains & Carbohydrates (Rice, Roti, Oats, Poha, Idli)",
        "serving_1_3": "1 child's cupped hand (~25-30g dry equivalent)",
        "serving_3_5": "1 to 1.5 child's cupped hands (~40-50g dry equivalent)",
        "icon": "🤲"
    },
    "Thumb": {
        "nutrient": "Healthy Fats & Oils (Desi Ghee, Butter, Nut Butter, Seeds)",
        "serving_1_3": "1 child's thumb (~1 tsp / 5g) per meal",
        "serving_3_5": "1 to 2 child's thumbs (~1.5 tsp / 7-10g) per meal",
        "icon": "👍"
    }
}

# --- Curated Meal Recipes by Category & Diet Preference ---
MEAL_DATABASE = {
    "1-3_years": {
        "Vegetarian": {
            "Breakfast": [
                {"name": "Soft Ragi Idli + Vegetable Sambar", "desc": "Steamed finger-millet cakes with mild lentil-veggie stew", "cal": 220, "p": 6.0, "c": 35.0, "f": 4.5, "hand_portion": "1 cupped hand idli + 1/2 palm sambar", "nutrients": "Calcium, Fiber, Iron"},
                {"name": "Moong Dal Cheela + Mint Curd Dip", "desc": "Golden yellow lentil pancake with fresh homemade curd", "cal": 210, "p": 7.5, "c": 28.0, "f": 5.0, "hand_portion": "1 palm size cheela + 1/2 fist curd", "nutrients": "High Protein, Zinc"},
                {"name": "Oats & Banana Porridge with Cow's Milk", "desc": "Creamy rolled oats cooked in milk with mashed banana & pinch of cardamom", "cal": 230, "p": 6.5, "c": 36.0, "f": 5.5, "hand_portion": "1 cupped hand porridge + 1 thumb ghee", "nutrients": "Energy, Vitamin B6, Calcium"}
            ],
            "Mid-Morning": [
                {"name": "Mashed Papaya / Stewed Apple + Roasted Makhana", "desc": "Soft seasonal fruit pieces with crushed ghee-roasted lotus seeds", "cal": 95, "p": 1.5, "c": 18.0, "f": 2.0, "hand_portion": "1 child fist fruit + 1/2 cupped hand makhana", "nutrients": "Vitamin A, Vitamin C, Antioxidants"},
                {"name": "Fresh Sweet Orange Slices + Crushed Almond Powder", "desc": "Juicy fiber-rich orange segments sprinkled with almond powder", "cal": 90, "p": 2.0, "c": 15.0, "f": 2.5, "hand_portion": "1 child fist fruit + 1 thumb nut powder", "nutrients": "Vitamin C, Healthy Fats"}
            ],
            "Lunch": [
                {"name": "Soft Dal Khichdi with Ghee + Carrot Beetroot Salad", "desc": "Comforting rice & yellow moong dal cooked soft with grated carrots and 1 tsp pure desi ghee", "cal": 320, "p": 9.5, "c": 48.0, "f": 8.0, "hand_portion": "1 cupped hand khichdi + 1 fist grated salad + 1 thumb ghee", "nutrients": "Complete Amino Acids, Iron, Beta Carotene"},
                {"name": "Paneer Bhurji + Soft Phulka + Spinach Dal", "desc": "Mildly spiced crumbled paneer with tender whole wheat phulka and iron-rich palak dal", "cal": 330, "p": 11.0, "c": 42.0, "f": 9.5, "hand_portion": "1 small phulka + 1/2 palm paneer + 1/2 fist spinach", "nutrients": "Calcium, Protein, Iron"}
            ],
            "Evening Snack": [
                {"name": "Warm Turmeric Milk (150ml) + 1 Whole Grain Biscuit", "desc": "Fortified warm cow's milk with organic turmeric & a whole wheat biscuit", "cal": 110, "p": 4.5, "c": 14.0, "f": 4.0, "hand_portion": "1 small cup milk + 1 child thumb biscuit", "nutrients": "Calcium, Immunity, Vitamin D"},
                {"name": "Curd Banana Smoothie with Jaggery", "desc": "Probiotic curd blended with ripe banana and a drop of organic jaggery", "cal": 120, "p": 4.0, "c": 20.0, "f": 3.0, "hand_portion": "1 small cup smoothie", "nutrients": "Probiotics, Potassium"}
            ],
            "Dinner": [
                {"name": "Vegetable Dalia Khichdi + Lauki (Bottle Gourd) Curry", "desc": "Broken wheat cooked with mixed vegetables and light bottle gourd sabzi", "cal": 240, "p": 6.5, "c": 38.0, "f": 5.0, "hand_portion": "1 cupped hand dalia + 1 fist lauki", "nutrients": "Easy Digestibility, Fiber"},
                {"name": "Soft Curd Rice with Pomegranate Kernels & Tadka", "desc": "Mashed soft rice with creamy homemade curd and ruby pomegranate seeds", "cal": 230, "p": 5.5, "c": 36.0, "f": 4.5, "hand_portion": "1 cupped hand curd rice + 1/2 fist pomegranate", "nutrients": "Gut Health, Vitamin C"}
            ]
        },
        "Eggetarian": {
            "Breakfast": [
                {"name": "Fluffy Scrambled Egg with Soft Toast Fingers", "desc": "1 farm egg scrambled in 1/2 tsp butter with whole wheat toast strips", "cal": 230, "p": 8.5, "c": 22.0, "f": 8.5, "hand_portion": "1 palm scrambled egg + 1 cupped hand toast", "nutrients": "High Biological Value Protein, Choline"},
                {"name": "Egg Dosa + Mild Coconut Chutney", "desc": "Crispy fermented rice pancake layered with 1 beaten egg", "cal": 240, "p": 8.0, "c": 28.0, "f": 7.5, "hand_portion": "1 small dosa with egg + 1 thumb chutney", "nutrients": "Protein, Vitamin B12, Energy"}
            ],
            "Mid-Morning": [
                {"name": "Mashed Papaya / Stewed Apple + Roasted Makhana", "desc": "Soft seasonal fruit pieces with crushed ghee-roasted lotus seeds", "cal": 95, "p": 1.5, "c": 18.0, "f": 2.0, "hand_portion": "1 child fist fruit + 1/2 cupped hand makhana", "nutrients": "Vitamin A, Vitamin C, Antioxidants"}
            ],
            "Lunch": [
                {"name": "Soft Dal Khichdi with Ghee + 1 Boiled Egg Slices", "desc": "Comforting rice & yellow moong dal with sliced boiled egg on top", "cal": 335, "p": 12.0, "c": 44.0, "f": 9.0, "hand_portion": "1 cupped hand khichdi + 1/2 palm egg + 1 thumb ghee", "nutrients": "Complete Amino Acids, Iron, Choline"}
            ],
            "Evening Snack": [
                {"name": "Warm Turmeric Milk (150ml) + 1 Whole Grain Biscuit", "desc": "Fortified warm cow's milk with organic turmeric & a whole wheat biscuit", "cal": 110, "p": 4.5, "c": 14.0, "f": 4.0, "hand_portion": "1 small cup milk + 1 child thumb biscuit", "nutrients": "Calcium, Immunity, Vitamin D"}
            ],
            "Dinner": [
                {"name": "Egg Drop Clear Vegetable Soup + Soft Phulka", "desc": "Warm clear vegetable soup with wisps of cooked egg and a soft buttered phulka", "cal": 240, "p": 8.0, "c": 32.0, "f": 6.5, "hand_portion": "1 small phulka + 1 fist soup", "nutrients": "Hydration, Protein, Vitamin A"}
            ]
        },
        "Non-Vegetarian": {
            "Breakfast": [
                {"name": "Fluffy Scrambled Egg with Soft Toast Fingers", "desc": "1 farm egg scrambled in 1/2 tsp butter with whole wheat toast strips", "cal": 230, "p": 8.5, "c": 22.0, "f": 8.5, "hand_portion": "1 palm scrambled egg + 1 cupped hand toast", "nutrients": "High Biological Value Protein, Choline"}
            ],
            "Mid-Morning": [
                {"name": "Mashed Papaya / Stewed Apple + Roasted Makhana", "desc": "Soft seasonal fruit pieces with crushed ghee-roasted lotus seeds", "cal": 95, "p": 1.5, "c": 18.0, "f": 2.0, "hand_portion": "1 child fist fruit + 1/2 cupped hand makhana", "nutrients": "Vitamin A, Vitamin C, Antioxidants"}
            ],
            "Lunch": [
                {"name": "Tender Minced Chicken & Veggie Stew + Steamed Rice", "desc": "Slow-cooked boneless minced chicken with carrots, potatoes, and soft rice", "cal": 340, "p": 13.5, "c": 42.0, "f": 8.5, "hand_portion": "1/2 palm minced chicken + 1 cupped hand rice + 1 fist veggies", "nutrients": "Bioavailable Iron, High Quality Protein, Zinc"}
            ],
            "Evening Snack": [
                {"name": "Warm Turmeric Milk (150ml) + 1 Whole Grain Biscuit", "desc": "Fortified warm cow's milk with organic turmeric & a whole wheat biscuit", "cal": 110, "p": 4.5, "c": 14.0, "f": 4.0, "hand_portion": "1 small cup milk + 1 child thumb biscuit", "nutrients": "Calcium, Immunity, Vitamin D"}
            ],
            "Dinner": [
                {"name": "Soft Fish Fillet Curry (Mild) + Steamed Rice", "desc": "Steamed boneless river fish cooked in light turmeric-tomato sauce with rice", "cal": 250, "p": 11.0, "c": 34.0, "f": 5.5, "hand_portion": "1/2 palm fish + 1 cupped hand rice", "nutrients": "DHA, Omega-3 Fatty Acids, Vitamin D"}
            ]
        }
    },
    "3-5_years": {
        "Vegetarian": {
            "Breakfast": [
                {"name": "Vegetable Stuffed Paratha (Paneer & Methi) + Curd", "desc": "Golden whole wheat flatbread stuffed with grated cottage cheese & fresh fenugreek", "cal": 290, "p": 9.5, "c": 40.0, "f": 8.5, "hand_portion": "1 child palm paratha + 1/2 fist curd", "nutrients": "Calcium, Iron, Vitamin A"},
                {"name": "Mixed Vegetable Poha with Peanuts & Lemon", "desc": "Flaked rice tempered with mustard, green peas, carrots, crunchy roasted peanuts, and vitamin C rich lemon", "cal": 280, "p": 7.0, "c": 44.0, "f": 7.5, "hand_portion": "1 cupped hand poha + 1 thumb peanuts", "nutrients": "Iron, Vitamin C, Healthy Fats"}
            ],
            "Mid-Morning": [
                {"name": "Seasonal Fruit Platter (Apple, Banana, Guava) + Soaked Walnuts", "desc": "Crisp fresh fruit cuts paired with brain-boosting soaked walnut halves", "cal": 130, "p": 2.5, "c": 24.0, "f": 4.0, "hand_portion": "1 fist mixed fruit + 1 thumb soaked walnuts", "nutrients": "Omega-3, Fiber, Vitamin C"}
            ],
            "Lunch": [
                {"name": "MyPlate Indian Thali: 2 Phulkas + Rajma Curry + Bhindi Sabzi + Cucumber Salad", "desc": "Kidney bean curry in mild tomato gravy, okra stir-fry, whole wheat phulkas, and crunchy cucumber sticks", "cal": 440, "p": 14.5, "c": 64.0, "f": 11.0, "hand_portion": "1 palm rajma + 1 fist bhindi & salad + 1 cupped hand (2 small phulkas)", "nutrients": "Plant Protein, Soluble Fiber, Potassium, Iron"},
                {"name": "Paneer Tikka Rice Bowl + Chana Dal + Steamed French Beans", "desc": "Grilled cottage cheese cubes, yellow bengal gram dal, brown rice, and crisp green beans", "cal": 450, "p": 16.0, "c": 60.0, "f": 12.0, "hand_portion": "1 palm paneer & dal + 1 fist beans + 1 cupped hand rice", "nutrients": "High Protein, Calcium, Zinc"}
            ],
            "Evening Snack": [
                {"name": "Sprouts & Sweet Corn Chaat + Fresh Buttermilk (Chaas)", "desc": "Lightly steamed sprouted moong & sweet corn with chaat masala and cumin buttermilk", "cal": 150, "p": 6.5, "c": 22.0, "f": 2.5, "hand_portion": "1 cupped hand chaat + 1 glass chaas", "nutrients": "Enzymes, Bioavailable Iron, Hydration"}
            ],
            "Dinner": [
                {"name": "Mixed Vegetable Dalia with Paneer Cubes + Roasted Papad", "desc": "Wholesome cracked wheat with carrots, peas, paneer and a crispy roasted papad", "cal": 320, "p": 10.5, "c": 46.0, "f": 8.0, "hand_portion": "1.5 cupped hands dalia + 1/2 palm paneer", "nutrients": "Complex Carbs, Lean Protein, Magnesium"}
            ]
        },
        "Eggetarian": {
            "Breakfast": [
                {"name": "2-Egg Veggie Omelette + Whole Grain Toast", "desc": "Whisked eggs folded with bell peppers, tomatoes, and spinach served with toasted whole wheat bread", "cal": 310, "p": 13.5, "c": 26.0, "f": 12.0, "hand_portion": "1 palm omelette + 1 cupped hand toast", "nutrients": "High Quality Protein, Choline, Vitamin B12"}
            ],
            "Mid-Morning": [
                {"name": "Seasonal Fruit Platter (Apple, Banana, Guava) + Soaked Walnuts", "desc": "Crisp fresh fruit cuts paired with brain-boosting soaked walnut halves", "cal": 130, "p": 2.5, "c": 24.0, "f": 4.0, "hand_portion": "1 fist mixed fruit + 1 thumb soaked walnuts", "nutrients": "Omega-3, Fiber, Vitamin C"}
            ],
            "Lunch": [
                {"name": "Egg Curry (2 Eggs) + Jeera Rice + Mixed Veggie Salad", "desc": "Hard-boiled eggs simmered in onion-tomato gravy with fragrant cumin rice and fresh salad", "cal": 450, "p": 15.0, "c": 56.0, "f": 14.0, "hand_portion": "1 palm egg curry + 1 cupped hand rice + 1 fist salad", "nutrients": "Protein, Lutein, Choline, Iron"}
            ],
            "Evening Snack": [
                {"name": "Sprouts & Sweet Corn Chaat + Fresh Buttermilk (Chaas)", "desc": "Lightly steamed sprouted moong & sweet corn with chaat masala and cumin buttermilk", "cal": 150, "p": 6.5, "c": 22.0, "f": 2.5, "hand_portion": "1 cupped hand chaat + 1 glass chaas", "nutrients": "Enzymes, Bioavailable Iron, Hydration"}
            ],
            "Dinner": [
                {"name": "Vegetable Egg Fried Rice (Low-Oil) + Dal Soup", "desc": "Steamed rice tossed with scrambled egg, shredded carrots, beans and a side of yellow dal soup", "cal": 340, "p": 11.5, "c": 48.0, "f": 9.0, "hand_portion": "1 cupped hand fried rice + 1/2 palm dal soup", "nutrients": "Balanced Macros, Fast Digestion"}
            ]
        },
        "Non-Vegetarian": {
            "Breakfast": [
                {"name": "2-Egg Veggie Omelette + Whole Grain Toast", "desc": "Whisked eggs folded with bell peppers, tomatoes, and spinach served with toasted whole wheat bread", "cal": 310, "p": 13.5, "c": 26.0, "f": 12.0, "hand_portion": "1 palm omelette + 1 cupped hand toast", "nutrients": "High Quality Protein, Choline, Vitamin B12"}
            ],
            "Mid-Morning": [
                {"name": "Seasonal Fruit Platter (Apple, Banana, Guava) + Soaked Walnuts", "desc": "Crisp fresh fruit cuts paired with brain-boosting soaked walnut halves", "cal": 130, "p": 2.5, "c": 24.0, "f": 4.0, "hand_portion": "1 fist mixed fruit + 1 thumb soaked walnuts", "nutrients": "Omega-3, Fiber, Vitamin C"}
            ],
            "Lunch": [
                {"name": "Homestyle Chicken Curry + 2 Phulkas + Cucumber Beetroot Salad", "desc": "Lean chicken pieces cooked in homestyle gravy with whole wheat phulkas and fresh salad", "cal": 460, "p": 19.0, "c": 52.0, "f": 13.0, "hand_portion": "1 palm chicken pieces + 1 cupped hand (2 phulkas) + 1 fist salad", "nutrients": "Heme Iron, High Biological Value Protein, Zinc, B6"}
            ],
            "Evening Snack": [
                {"name": "Sprouts & Sweet Corn Chaat + Fresh Buttermilk (Chaas)", "desc": "Lightly steamed sprouted moong & sweet corn with chaat masala and cumin buttermilk", "cal": 150, "p": 6.5, "c": 22.0, "f": 2.5, "hand_portion": "1 cupped hand chaat + 1 glass chaas", "nutrients": "Enzymes, Bioavailable Iron, Hydration"}
            ],
            "Dinner": [
                {"name": "Grilled Pomfret / Rohu Fish + Steamed Rice + Palak Dal", "desc": "Mildly spiced pan-grilled fresh fish with soft steamed rice and spinach dal", "cal": 340, "p": 16.0, "c": 44.0, "f": 8.5, "hand_portion": "1 palm fish + 1 cupped hand rice + 1/2 palm dal", "nutrients": "DHA, EPA, Vitamin D, Selenium"}
            ]
        }
    }
}

# --- Clinical Nutrition Boosters by Growth Status ---
GROWTH_STATUS_ADVICE = {
    "Underweight": {
        "title": "Healthy Weight Gain & Calorie Booster Plan",
        "focus": "Increase caloric density without increasing bulk. Add energy boosters to regular meals.",
        "tips": [
            "Add 1 tsp pure desi ghee or butter to warm khichdi, dal, or porridge.",
            "Use almond / cashew nut powder in daily milk, pancakes, or halwa.",
            "Offer 5-6 small nutrient-dense meals instead of 3 large meals.",
            "Avoid giving water 30 minutes before meals to prevent premature satiety.",
            "Include whole milk, paneer, eggs, and ripe bananas daily."
        ],
        "calorie_adj": +150
    },
    "Stunted": {
        "title": "Linear Growth & Height Maximization Plan",
        "focus": "Prioritize high-bioavailability proteins, Calcium, Zinc, and Vitamin D for bone growth.",
        "tips": [
            "Ensure a high-quality protein source at EVERY meal (Egg, Paneer, Dal, Fish, Milk).",
            "Incorporate ragi (finger millet) and sesame seeds for dense calcium.",
            "Pair iron-rich foods (spinach, lentils) with vitamin C (lemon, amla, orange) for enhanced absorption.",
            "Encourage 30+ minutes of morning outdoor play for natural Vitamin D synthesis.",
            "Maintain consistent 10-12 hours of uninterrupted night sleep for growth hormone release."
        ],
        "calorie_adj": +50
    },
    "Stunted & Underweight": {
        "title": "Comprehensive Catch-Up Growth Protocol",
        "focus": "Combined high-protein, calorie-dense, and micronutrient-fortified dietary rehabilitation.",
        "tips": [
            "Intensive nutrient pacing: 3 solid meals + 3 rich snacks daily.",
            "Pair protein + healthy fats in every offering (e.g. egg + butter, dalia + ghee + paneer).",
            "Include daily dark green leafy vegetables, orange vegetables (carrots), and citrus fruits.",
            "Consult your pediatrician for targeted multi-micronutrient supplementation (Zinc, Iron, Vitamin A/D)."
        ],
        "calorie_adj": +200
    },
    "Overweight": {
        "title": "Active Metabolism & Portion Awareness Plan",
        "focus": "Improve food quality and nutrient density; NEVER place a young child on a restrictive diet.",
        "tips": [
            "Follow the 50% Plate Rule: Half of lunch and dinner plates should be vegetables and salad.",
            "Replace packaged juices, sodas, and sweet biscuits with whole fruits and roasted seeds.",
            "Serve meals on smaller, colorful plates and let the child eat slowly without screens.",
            "Ensure 60+ minutes of active, heart-pumping outdoor play every single day.",
            "Encourage drinking plain water instead of sugary milk beverages."
        ],
        "calorie_adj": -100
    },
    "Obese": {
        "title": "Pediatric Health & Lifestyle Rebalancing Protocol",
        "focus": "Gentle weight stabilization allowing height to catch up under pediatric supervision.",
        "tips": [
            "Prioritize lean proteins and high-fiber legumes (chana, rajma, sprouts) to maintain satiety.",
            "Strictly eliminate ultra-processed snacks, deep-fried snacks, and hidden sugars.",
            "Family-wide healthy eating habits — children model parents' food behaviors.",
            "Cap screen time to strictly under 1 hour per day; replace with active park play.",
            "Schedule regular height-weight velocity checks with your pediatrician."
        ],
        "calorie_adj": -150
    },
    "Healthy": {
        "title": "Balanced Growth & Immune Fortification Plan",
        "focus": "Maintain diverse, colorful, and joyful eating habits across all five food groups.",
        "tips": [
            "Offer a rainbow plate with 5 different natural colors of fruits and vegetables daily.",
            "Maintain fixed meal and snack schedules to establish a lifelong healthy circadian rhythm.",
            "Involve your child in washing veggies and setting the table to foster positive food connection.",
            "Ensure proper hydration with water and buttermilk throughout active play."
        ],
        "calorie_adj": 0
    }
}

def generate_custom_meal_plan(age_months: float, sex: str, diet_pref: str, status: str = "Healthy", activity_level: str = "Moderate"):
    """
    Generate a full-day, scientifically calibrated 5-meal plan tailored to child's exact profile.
    """
    age_group = "1-3_years" if age_months < 36 else "3-5_years"
    diet = diet_pref if diet_pref in ["Vegetarian", "Eggetarian", "Non-Vegetarian"] else "Vegetarian"
    
    # Base RDA
    base_rda = RDA_TABLE[age_group].copy()
    
    # Status adjustment
    status_key = status if status in GROWTH_STATUS_ADVICE else "Healthy"
    status_info = GROWTH_STATUS_ADVICE[status_key]
    
    # Activity multiplier
    act_mult = 1.0
    if activity_level == "Sedentary":
        act_mult = 0.95
    elif activity_level == "Active":
        act_mult = 1.08
    
    adj_calories = int(base_rda["calories"] * act_mult + status_info["calorie_adj"])
    
    # Select meals from database
    db_group = MEAL_DATABASE.get(age_group, MEAL_DATABASE["1-3_years"]).get(diet, MEAL_DATABASE["1-3_years"]["Vegetarian"])
    
    meals_selected = {}
    total_cal = 0
    total_p = 0.0
    total_c = 0.0
    total_f = 0.0
    
    for meal_type in ["Breakfast", "Mid-Morning", "Lunch", "Evening Snack", "Dinner"]:
        options = db_group.get(meal_type, [])
        if options:
            chosen = options[0]  # default to primary balanced option
            meals_selected[meal_type] = chosen
            total_cal += chosen["cal"]
            total_p += chosen["p"]
            total_c += chosen["c"]
            total_f += chosen["f"]
            
    return {
        "age_months": age_months,
        "age_group": age_group,
        "sex": sex,
        "diet_pref": diet,
        "status": status_key,
        "activity_level": activity_level,
        "target_rda": {
            "calories": adj_calories,
            "protein_g": base_rda["protein_g"],
            "fat_g": base_rda["fat_g"],
            "carbs_g": base_rda["carbs_g"],
            "calcium_mg": base_rda["calcium_mg"],
            "iron_mg": base_rda["iron_mg"],
            "water_ml": base_rda["water_ml"]
        },
        "plan_totals": {
            "calories": total_cal,
            "protein_g": round(total_p, 1),
            "carbs_g": round(total_c, 1),
            "fat_g": round(total_f, 1)
        },
        "meals": meals_selected,
        "clinical_advice": status_info,
        "hand_portions": HAND_PORTION_GUIDE
    }

def create_meal_plan_pdf(child_name: str, plan: dict) -> BytesIO:
    """Generate a high-quality PDF meal plan chart for parents."""
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    margin_left = 1.5 * cm
    content_width = width - 3.0 * cm
    
    # Header Banner
    c.setFillColor(colors.HexColor("#0284c7"))
    c.rect(0, height - 1.4 * cm, width, 1.4 * cm, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(margin_left, height - 0.95 * cm, "AI Child Daily Meal & Nutrition Plan")
    
    # Subheader & Metadata
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 12)
    y = height - 2.0 * cm
    c.drawString(margin_left, y, f"Child Name: {child_name}")
    c.setFont("Helvetica", 10)
    age_y = int(plan['age_months'] // 12)
    age_m = int(plan['age_months'] % 12)
    c.drawString(margin_left + content_width / 2, y, f"Age: {age_y}y {age_m}m | Diet: {plan['diet_pref']}")
    
    y -= 0.6 * cm
    c.setFont("Helvetica-Bold", 10)
    c.setFillColor(colors.HexColor("#0369a1"))
    c.drawString(margin_left, y, f"Target Daily Calories: ~{plan['target_rda']['calories']} kcal | Status Focus: {plan['status']}")
    
    y -= 0.7 * cm
    c.setStrokeColor(colors.HexColor("#cbd5e1"))
    c.setLineWidth(0.8)
    c.line(margin_left, y, margin_left + content_width, y)
    
    # 5 Meals Section
    y -= 0.8 * cm
    c.setFillColor(colors.HexColor("#0f172a"))
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin_left, y, "Daily 5-Meal Schedule")
    
    y -= 0.6 * cm
    for meal_name, m_data in plan["meals"].items():
        if y < 4.0 * cm:
            c.showPage()
            y = height - 2.0 * cm
            
        c.setFillColor(colors.HexColor("#f8fafc"))
        c.roundRect(margin_left, y - 1.1 * cm, content_width, 1.2 * cm, 4, fill=1, stroke=1)
        
        c.setFillColor(colors.HexColor("#0284c7"))
        c.setFont("Helvetica-Bold", 10)
        c.drawString(margin_left + 0.3 * cm, y - 0.2 * cm, f"{meal_name}: {m_data['name']}")
        
        c.setFillColor(colors.HexColor("#475569"))
        c.setFont("Helvetica", 8.5)
        c.drawString(margin_left + 0.3 * cm, y - 0.55 * cm, f"{m_data['desc']}")
        c.drawString(margin_left + 0.3 * cm, y - 0.9 * cm, f"Portion: {m_data['hand_portion']} | Nutrients: {m_data['nutrients']} | ~{m_data['cal']} kcal")
        
        y -= 1.45 * cm
        
    # Clinical Nutritional Tips
    if y > 4.5 * cm:
        y -= 0.3 * cm
        c.setFillColor(colors.HexColor("#0f172a"))
        c.setFont("Helvetica-Bold", 11)
        c.drawString(margin_left, y, f"Nutritional Guidance ({plan['clinical_advice']['title']})")
        
        y -= 0.5 * cm
        c.setFont("Helvetica", 8.5)
        c.setFillColor(colors.HexColor("#334155"))
        for tip in plan['clinical_advice']['tips'][:4]:
            c.drawString(margin_left + 0.3 * cm, y, f"• {tip}")
            y -= 0.45 * cm
            
    # Footer
    c.setFont("Helvetica-Oblique", 7.5)
    c.setFillColor(colors.HexColor("#64748b"))
    c.drawString(margin_left, 1.0 * cm, "Source: ICMR-NIN Dietary Guidelines 2024, AskNestle Meal Plan, WHO Healthy Diet Guidelines. Informational only.")
    
    c.save()
    buffer.seek(0)
    return buffer
