import csv
import os
import random
import math

# Set random seed for reproducibility
random.seed(42)

DISTRICT_DATA = {
    "Peshawar": {"division": "Peshawar", "terrain": "Plain", "base_literacy": 72.0, "base_income": 65000, "weight": 1.2},
    "Mardan": {"division": "Mardan", "terrain": "Plain", "base_literacy": 65.0, "base_income": 50000, "weight": 1.0},
    "Swabi": {"division": "Mardan", "terrain": "Plain", "base_literacy": 68.0, "base_income": 52000, "weight": 0.9},
    "Charsadda": {"division": "Peshawar", "terrain": "Plain", "base_literacy": 61.0, "base_income": 45000, "weight": 0.8},
    "Nowshera": {"division": "Peshawar", "terrain": "Plain", "base_literacy": 64.0, "base_income": 48000, "weight": 0.8},
    "Swat": {"division": "Malakand", "terrain": "Mountainous", "base_literacy": 62.0, "base_income": 42000, "weight": 1.0},
    "Chitral Lower": {"division": "Malakand", "terrain": "Mountainous", "base_literacy": 75.0, "base_income": 38000, "weight": 0.4},
    "Chitral Upper": {"division": "Malakand", "terrain": "Mountainous", "base_literacy": 78.0, "base_income": 36000, "weight": 0.3},
    "Dir Lower": {"division": "Malakand", "terrain": "Mountainous", "base_literacy": 58.0, "base_income": 38000, "weight": 0.7},
    "Dir Upper": {"division": "Malakand", "terrain": "Mountainous", "base_literacy": 52.0, "base_income": 34000, "weight": 0.6},
    "Shangla": {"division": "Malakand", "terrain": "Mountainous", "base_literacy": 42.0, "base_income": 28000, "weight": 0.4},
    "Buner": {"division": "Malakand", "terrain": "Hilly", "base_literacy": 51.0, "base_income": 35000, "weight": 0.5},
    "Abbottabad": {"division": "Hazara", "terrain": "Hilly", "base_literacy": 76.0, "base_income": 60000, "weight": 0.8},
    "Haripur": {"division": "Hazara", "terrain": "Plain", "base_literacy": 72.0, "base_income": 58000, "weight": 0.7},
    "Mansehra": {"division": "Hazara", "terrain": "Mountainous", "base_literacy": 63.0, "base_income": 40000, "weight": 0.8},
    "Battagram": {"division": "Hazara", "terrain": "Mountainous", "base_literacy": 46.0, "base_income": 30000, "weight": 0.3},
    "Kohistan Upper": {"division": "Hazara", "terrain": "Mountainous", "base_literacy": 25.0, "base_income": 22000, "weight": 0.2},
    "Kohistan Lower": {"division": "Hazara", "terrain": "Mountainous", "base_literacy": 26.0, "base_income": 23000, "weight": 0.2},
    "Kolai Pallas": {"division": "Hazara", "terrain": "Mountainous", "base_literacy": 22.0, "base_income": 21000, "weight": 0.15},
    "Torghar": {"division": "Hazara", "terrain": "Mountainous", "base_literacy": 31.0, "base_income": 24000, "weight": 0.15},
    "Kohat": {"division": "Kohat", "terrain": "Hilly", "base_literacy": 62.0, "base_income": 46000, "weight": 0.7},
    "Karak": {"division": "Kohat", "terrain": "Hilly", "base_literacy": 65.0, "base_income": 43000, "weight": 0.6},
    "Hangu": {"division": "Kohat", "terrain": "Hilly", "base_literacy": 52.0, "base_income": 35000, "weight": 0.4},
    "Bannu": {"division": "Bannu", "terrain": "Plain", "base_literacy": 56.0, "base_income": 38000, "weight": 0.7},
    "Lakki Marwat": {"division": "Bannu", "terrain": "Plain", "base_literacy": 49.0, "base_income": 32000, "weight": 0.6},
    "Dera Ismail Khan": {"division": "D.I. Khan", "terrain": "Plain", "base_literacy": 54.0, "base_income": 42000, "weight": 0.8},
    "Tank": {"division": "D.I. Khan", "terrain": "Plain", "base_literacy": 43.0, "base_income": 28000, "weight": 0.3},
    "Khyber": {"division": "Peshawar", "terrain": "Mountainous", "base_literacy": 49.0, "base_income": 35000, "weight": 0.6},
    "Kurram": {"division": "Kohat", "terrain": "Mountainous", "base_literacy": 55.0, "base_income": 37000, "weight": 0.5},
    "Bajaur": {"division": "Malakand", "terrain": "Mountainous", "base_literacy": 36.0, "base_income": 27000, "weight": 0.6},
    "Mohmand": {"division": "Peshawar", "terrain": "Mountainous", "base_literacy": 33.0, "base_income": 26000, "weight": 0.4},
    "Orakzai": {"division": "Kohat", "terrain": "Mountainous", "base_literacy": 39.0, "base_income": 28000, "weight": 0.3},
    "North Waziristan": {"division": "Bannu", "terrain": "Mountainous", "base_literacy": 36.0, "base_income": 29000, "weight": 0.4},
    "South Waziristan Upper": {"division": "D.I. Khan", "terrain": "Mountainous", "base_literacy": 29.0, "base_income": 25000, "weight": 0.3},
    "South Waziristan Lower": {"division": "D.I. Khan", "terrain": "Mountainous", "base_literacy": 31.0, "base_income": 26000, "weight": 0.3}
}

DISTRICT_CROPS = {
    "Peshawar": ["Wheat", "Maize", "Sugarcane", "Rice"],
    "Mardan": ["Wheat", "Maize", "Sugarcane", "Tobacco"],
    "Swabi": ["Wheat", "Maize", "Tobacco"],
    "Charsadda": ["Wheat", "Maize", "Sugarcane", "Rice"],
    "Nowshera": ["Wheat", "Maize", "Sugarcane"],
    "Swat": ["Apples", "Peaches", "Wheat", "Rice"],
    "Chitral Lower": ["Apples", "Peaches", "Wheat"],
    "Chitral Upper": ["Apples", "Wheat"],
    "Dir Lower": ["Apples", "Peaches", "Wheat"],
    "Dir Upper": ["Apples", "Peaches", "Wheat"],
    "Shangla": ["Maize", "Wheat"],
    "Buner": ["Maize", "Wheat", "Peaches"],
    "Abbottabad": ["Maize", "Wheat", "Peaches"],
    "Haripur": ["Wheat", "Maize", "Peaches"],
    "Mansehra": ["Maize", "Wheat", "Apples", "Peaches"],
    "Battagram": ["Maize", "Wheat"],
    "Kohistan Upper": ["Wheat", "Maize"],
    "Kohistan Lower": ["Wheat", "Maize"],
    "Kolai Pallas": ["Wheat", "Maize"],
    "Torghar": ["Wheat", "Maize"],
    "Kohat": ["Wheat", "Maize"],
    "Karak": ["Wheat", "Gram"],
    "Hangu": ["Wheat", "Maize"],
    "Bannu": ["Wheat", "Maize", "Sugarcane"],
    "Lakki Marwat": ["Wheat", "Gram"],
    "Dera Ismail Khan": ["Wheat", "Sugarcane", "Rice", "Dates"],
    "Tank": ["Wheat", "Gram"],
    "Khyber": ["Wheat", "Maize"],
    "Kurram": ["Wheat", "Maize", "Apples", "Peaches"],
    "Bajaur": ["Wheat", "Maize"],
    "Mohmand": ["Wheat", "Maize"],
    "Orakzai": ["Wheat", "Maize"],
    "North Waziristan": ["Wheat", "Maize"],
    "South Waziristan Upper": ["Wheat", "Apples", "Peaches"],
    "South Waziristan Lower": ["Wheat", "Maize"]
}

districts = list(DISTRICT_DATA.keys())
weights = [d["weight"] for d in DISTRICT_DATA.values()]

def generate_development_dataset(filename, num_rows):
    print(f"Generating development dataset: {filename} with {num_rows} records...")
    headers = [
        "community_id", "district", "division", "terrain_type", "population",
        "literacy_rate", "school_enrollment_rate", "health_facility_distance_km",
        "doctors_per_1000", "road_connectivity_index", "electricity_access_pct",
        "clean_water_access_pct", "internet_penetration_pct", "average_household_income_pkr",
        "public_funding_allocated_million_pkr", "development_score"
    ]
    
    with open(filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for i in range(1, num_rows + 1):
            comm_id = f"COM-{i:06d}"
            dist = random.choices(districts, weights=weights)[0]
            dist_info = DISTRICT_DATA[dist]
            div = dist_info["division"]
            terrain = dist_info["terrain"]
            
            # Population (log-normal, representing villages and small towns)
            population = int(random.lognormvariate(math.log(3000), 0.6))
            population = max(500, min(population, 50000))
            
            # Literacy rate (base +/- random variance)
            lit_base = dist_info["base_literacy"]
            literacy = lit_base + random.gauss(0, 7)
            literacy = max(10.0, min(literacy, 98.0))
            
            # School enrollment (strongly correlated with literacy)
            enrollment = literacy * 1.08 + random.gauss(0, 4)
            enrollment = max(15.0, min(enrollment, 99.0))
            
            # Distance to nearest health facility (km)
            if terrain == "Mountainous":
                dist_mean = 22.0
            elif terrain == "Hilly":
                dist_mean = 11.0
            else:
                dist_mean = 4.0
            health_dist = random.expovariate(1.0 / dist_mean)
            health_dist = max(0.1, min(health_dist, 60.0))
            
            # Doctors per 1000 people
            doc_mean = (dist_info["base_income"] / 55000) * 0.7
            doctors = random.gammavariate(2, doc_mean/2)
            doctors = max(0.01, min(doctors, 7.0))
            
            # Road connectivity index
            if terrain == "Mountainous":
                road_base = 35.0
            elif terrain == "Hilly":
                road_base = 62.0
            else:
                road_base = 82.0
            road_index = road_base + random.gauss(0, 9)
            road_index = max(5.0, min(road_index, 100.0))
            
            # Utilities access percentage
            elec_base = 96.0 if terrain == "Plain" else (68.0 if terrain == "Hilly" else 42.0)
            electricity = elec_base + random.gauss(0, 10)
            electricity = max(0.0, min(electricity, 100.0))
            
            water_base = 78.0 if terrain == "Plain" else (64.0 if terrain == "Hilly" else 50.0)
            clean_water = water_base + random.gauss(0, 12)
            clean_water = max(5.0, min(clean_water, 100.0))
            
            internet_base = 62.0 if terrain == "Plain" else (42.0 if terrain == "Hilly" else 22.0)
            internet = internet_base + (dist_info["base_income"] - 40000) / 1100 + random.gauss(0, 9)
            internet = max(2.0, min(internet, 98.0))
            
            # Household income
            income_base = dist_info["base_income"]
            income = income_base + (literacy - lit_base) * 750 + random.gauss(0, 7500)
            income = max(8000.0, income)
            
            # Public development funding (millions PKR)
            funding = random.uniform(0.1, 12.0)
            if random.random() < 0.12:  # 12% probability of a major development project
                funding += random.uniform(15.0, 75.0)
                
            # Normalization and target score calculation (0 to 100)
            x_lit = literacy / 100.0
            x_enr = enrollment / 100.0
            x_hdist = (60.0 - health_dist) / 60.0
            x_docs = min(doctors / 3.0, 1.0)
            x_road = road_index / 100.0
            x_elec = electricity / 100.0
            x_water = clean_water / 100.0
            x_net = internet / 100.0
            x_inc = min(income / 110000.0, 1.0)
            x_fund = min(funding / 45.0, 1.0)
            
            raw_score = (
                0.15 * x_lit +
                0.12 * x_enr +
                0.10 * x_hdist +
                0.08 * x_docs +
                0.12 * x_road +
                0.08 * x_elec +
                0.08 * x_water +
                0.07 * x_net +
                0.12 * x_inc +
                0.08 * x_fund
            ) * 100.0
            
            # Penalty for terrain constraints
            if terrain == "Mountainous":
                raw_score -= 5.0
            elif terrain == "Hilly":
                raw_score -= 2.5
                
            dev_score = raw_score + random.gauss(0, 2.0)
            dev_score = max(0.0, min(dev_score, 100.0))
            
            # Formats
            population = int(population)
            literacy = round(literacy, 2)
            enrollment = round(enrollment, 2)
            health_dist = round(health_dist, 2)
            doctors = round(doctors, 3)
            road_index = round(road_index, 2)
            electricity = round(electricity, 2)
            clean_water = round(clean_water, 2)
            internet = round(internet, 2)
            income = round(income, 0)
            funding = round(funding, 3)
            dev_score = round(dev_score, 2)
            
            writer.writerow([
                comm_id, dist, div, terrain, population,
                literacy, enrollment, health_dist,
                doctors, road_index, electricity,
                clean_water, internet, income,
                funding, dev_score
            ])
    print(f"Finished generating {filename}.")

def generate_agricultural_dataset(filename, num_rows):
    print(f"Generating agricultural dataset: {filename} with {num_rows} records...")
    headers = [
        "farm_id", "district", "crop_type", "farm_size_acres", "elevation_meters",
        "soil_ph", "fertilizer_used_bags_per_acre", "organic_farming", "irrigation_type",
        "annual_rainfall_mm", "average_temperature_celsius", "pesticide_sprays_count",
        "mechanization_level", "distance_to_mandi_km", "crop_yield_tons_per_acre"
    ]
    
    base_yields = {
        "Wheat": 1.4,
        "Maize": 2.1,
        "Sugarcane": 24.0,
        "Tobacco": 0.95,
        "Rice": 1.7,
        "Peaches": 4.5,
        "Apples": 5.2,
        "Gram": 0.42,
        "Dates": 3.2
    }
    
    with open(filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for i in range(1, num_rows + 1):
            farm_id = f"FARM-{i:06d}"
            dist = random.choices(districts, weights=weights)[0]
            dist_info = DISTRICT_DATA[dist]
            terrain = dist_info["terrain"]
            div = dist_info["division"]
            
            available_crops = DISTRICT_CROPS.get(dist, ["Wheat", "Maize"])
            crop = random.choice(available_crops)
            
            # Farm size in acres
            farm_size = random.lognormvariate(math.log(3.5), 0.75)
            farm_size = max(0.5, min(farm_size, 120.0))
            
            # Elevation based on terrain
            if terrain == "Mountainous":
                elevation = random.uniform(1100, 2600)
            elif terrain == "Hilly":
                elevation = random.uniform(550, 1100)
            else:
                elevation = random.uniform(180, 550)
                
            # Soil pH
            soil_ph = random.gauss(7.0, 0.45)
            soil_ph = max(4.8, min(soil_ph, 9.2))
            
            # Organic farming
            organic_prob = 0.28 if terrain == "Mountainous" else 0.07
            organic = "Yes" if random.random() < organic_prob else "No"
            
            # Fertilizers
            if organic == "Yes":
                fertilizer = 0.0
            else:
                fert_mean = 5.0
                if crop == "Sugarcane":
                    fert_mean = 9.0
                elif crop == "Tobacco":
                    fert_mean = 6.5
                elif crop == "Gram":
                    fert_mean = 1.5
                fertilizer = random.gauss(fert_mean, 1.3)
                fertilizer = max(0.0, min(fertilizer, 15.0))
                
            # Irrigation
            if terrain == "Mountainous":
                irrigation_options = ["Rain-fed", "Canal", "Drip-irrigation"]
                irrigation_weights = [0.65, 0.30, 0.05]
            elif terrain == "Hilly":
                irrigation_options = ["Rain-fed", "Canal", "Tube-well", "Drip-irrigation"]
                irrigation_weights = [0.42, 0.28, 0.26, 0.04]
            else:
                irrigation_options = ["Canal", "Tube-well", "Rain-fed", "Drip-irrigation"]
                irrigation_weights = [0.58, 0.32, 0.08, 0.02]
            irrigation = random.choices(irrigation_options, weights=irrigation_weights)[0]
            
            # Rainfall
            if div == "Hazara":
                rain_mean = 1150.0
            elif div == "Malakand":
                rain_mean = 800.0
            elif div in ["Peshawar", "Mardan"]:
                rain_mean = 460.0
            elif div == "Kohat":
                rain_mean = 340.0
            else:
                rain_mean = 260.0
            rainfall = random.gauss(rain_mean, rain_mean * 0.12)
            rainfall = max(40.0, rainfall)
            
            # Temperature
            if elevation > 1600:
                temp_mean = 15.5
            elif elevation > 900:
                temp_mean = 22.0
            else:
                temp_mean = 28.5 if div != "D.I. Khan" else 33.0
            temp = random.gauss(temp_mean, 2.2)
            
            # Pesticide sprays
            if organic == "Yes":
                pesticides = 0
            else:
                pest_base = 2
                if crop in ["Apples", "Peaches", "Tobacco", "Sugarcane"]:
                    pest_base = 4
                elif crop == "Gram":
                    pest_base = 1
                pesticides = int(random.gauss(pest_base, 0.95))
                pesticides = max(0, min(pesticides, 8))
                
            # Mechanization
            if terrain == "Mountainous":
                mech_mean = 22.0
            elif terrain == "Hilly":
                mech_mean = 48.0
            else:
                mech_mean = 72.0
            mechanization = mech_mean + random.gauss(0, 10.0)
            mechanization = max(5.0, min(mechanization, 100.0))
            
            # Mandi distance
            mandi_mean = 24.0 if terrain == "Mountainous" else (14.0 if terrain == "Hilly" else 8.0)
            dist_mandi = random.expovariate(1.0 / mandi_mean)
            dist_mandi = max(0.5, min(dist_mandi, 75.0))
            
            # Multipliers logic
            base = base_yields[crop]
            
            ph_diff = abs(soil_ph - 6.8)
            ph_mult = max(0.65, 1.0 - 0.18 * (ph_diff ** 1.4))
            
            if fertilizer == 0:
                fert_mult = 0.68 if organic == "No" else 0.96
            else:
                fert_mult = 0.8 + 0.082 * fertilizer - 0.0031 * (fertilizer ** 2)
                fert_mult = max(0.5, min(fert_mult, 1.45))
                
            if irrigation == "Canal":
                irr_mult = 1.14
            elif irrigation == "Tube-well":
                irr_mult = 1.09
            elif irrigation == "Drip-irrigation":
                irr_mult = 1.26
            else:
                rain_factor = min(rainfall / 500.0, 1.15)
                irr_mult = 0.48 * rain_factor + 0.32
                
            climate_mult = 1.0
            if crop in ["Apples", "Peaches"]:
                if temp > 22.0:
                    climate_mult *= 0.55
                elif temp < 10.0:
                    climate_mult *= 0.82
                else:
                    climate_mult *= (1.0 + 0.05 * (18 - abs(temp - 16)))
            elif crop == "Sugarcane":
                if temp < 20.0:
                    climate_mult *= 0.48
                else:
                    climate_mult *= min(1.3, 0.72 + 0.015 * temp)
                    
            if organic == "No":
                pest_mult = 0.84 + 0.075 * pesticides - 0.0085 * (pesticides ** 2)
                pest_mult = max(0.68, min(pest_mult, 1.18))
            else:
                pest_mult = 1.0
                
            mech_mult = 0.88 + 0.0028 * mechanization
            
            yield_val = base * ph_mult * fert_mult * irr_mult * climate_mult * pest_mult * mech_mult
            
            # Gaussian Noise (approx 7.5%)
            noise = random.gauss(0, yield_val * 0.075)
            final_yield = yield_val + noise
            final_yield = max(base * 0.12, final_yield)
            
            # Formats
            farm_size = round(farm_size, 2)
            elevation = round(elevation, 1)
            soil_ph = round(soil_ph, 2)
            fertilizer = round(fertilizer, 1)
            rainfall = round(rainfall, 1)
            temp = round(temp, 1)
            mechanization = round(mechanization, 1)
            dist_mandi = round(dist_mandi, 2)
            final_yield = round(final_yield, 3)
            
            writer.writerow([
                farm_id, dist, crop, farm_size, elevation,
                soil_ph, fertilizer, organic, irrigation,
                rainfall, temp, pesticides,
                mechanization, dist_mandi, final_yield
            ])
    print(f"Finished generating {filename}.")

def generate_infrastructure_dataset(filename, num_rows):
    print(f"Generating infrastructure dataset: {filename} with {num_rows} records...")
    sectors = [
        "Roads & Bridges", "Education (Schools/Colleges)", "Health (BHUs/Hospitals)",
        "Water Supply & Sanitation", "Energy & Power", "Tourism & Sports", "Agriculture & Forestry"
    ]
    sector_weights = [0.28, 0.22, 0.18, 0.12, 0.08, 0.06, 0.06]
    
    headers = [
        "project_id", "project_sector", "district", "terrain_complexity", "project_scale",
        "estimated_duration_months", "approved_cost_million_pkr", "contractor_experience_years",
        "distance_from_hq_km", "funding_release_rate_pct", "monitoring_frequency_per_year",
        "year_of_initiation", "inflation_rate_at_start_pct", "labor_availability_index",
        "actual_cost_million_pkr", "actual_duration_months"
    ]
    
    with open(filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for i in range(1, num_rows + 1):
            project_id = f"PRJ-{i:06d}"
            sector = random.choices(sectors, weights=sector_weights)[0]
            dist = random.choices(districts, weights=weights)[0]
            dist_info = DISTRICT_DATA[dist]
            terrain = dist_info["terrain"]
            
            if terrain == "Mountainous":
                complexity = "High"
            elif terrain == "Hilly":
                complexity = random.choices(["Medium", "High"], weights=[0.72, 0.28])[0]
            else:
                complexity = random.choices(["Low", "Medium"], weights=[0.88, 0.12])[0]
                
            scale = random.choices(["Small", "Medium", "Large", "Mega"], weights=[0.48, 0.36, 0.13, 0.03])[0]
            
            if scale == "Small":
                cost = random.uniform(2.5, 22.0)
            elif scale == "Medium":
                cost = random.uniform(22.0, 110.0)
            elif scale == "Large":
                cost = random.uniform(110.0, 520.0)
            else:
                cost = random.uniform(520.0, 2800.0)
                
            if scale == "Small":
                est_duration = random.uniform(6, 12)
            elif scale == "Medium":
                est_duration = random.uniform(12, 24)
            elif scale == "Large":
                est_duration = random.uniform(24, 38)
            else:
                est_duration = random.uniform(38, 60)
                
            if sector in ["Roads & Bridges", "Energy & Power"]:
                est_duration *= 1.25
            elif sector in ["Water Supply & Sanitation", "Agriculture & Forestry"]:
                est_duration *= 0.85
            est_duration = int(round(est_duration))
            
            contractor_exp = int(random.triangular(1, 25, 6))
            
            dist_hq = random.uniform(0.5, 95.0)
            if complexity == "High":
                dist_hq += random.uniform(10, 45)
            dist_hq = max(0.5, dist_hq)
            
            funding_rate = random.gauss(76.0, 14.0)
            funding_rate = max(20.0, min(funding_rate, 100.0))
            
            monitoring = int(random.uniform(0, 12))
            year = random.randint(2015, 2025)
            
            # Historical inflation in Pakistan
            if year in [2022, 2023, 2024]:
                inflation = random.uniform(18.0, 30.0)
            elif year in [2015, 2016, 2017, 2018]:
                inflation = random.uniform(3.5, 7.0)
            else:
                inflation = random.uniform(7.5, 12.5)
                
            if complexity == "High":
                labor_base = 42.0
            else:
                labor_base = 76.0
            labor_index = labor_base + random.gauss(0, 9.5)
            labor_index = max(10.0, min(labor_index, 100.0))
            
            comp_factor = 1.0 if complexity == "Low" else (1.14 if complexity == "Medium" else 1.45)
            fund_factor = 1.0 + (100.0 - funding_rate) * 0.0095
            contractor_factor = 1.12 - 0.009 * min(contractor_exp, 18)
            dist_factor = 1.0 + (dist_hq * 0.0019)
            labor_factor = 1.22 - 0.0028 * labor_index
            
            duration_mult = comp_factor * fund_factor * contractor_factor * dist_factor * labor_factor
            actual_duration = est_duration * duration_mult
            
            duration_noise = random.gauss(0, actual_duration * 0.08)
            actual_duration += duration_noise
            actual_duration = max(est_duration * 0.9, actual_duration)
            actual_duration = int(round(actual_duration))
            
            dur_overrun_ratio = (actual_duration / est_duration) - 1.0
            cost_dur_factor = 1.0 + 0.32 * max(0.0, dur_overrun_ratio)
            
            inflation_factor = 1.0 + (inflation / 100.0) * (actual_duration / 12.0) * 0.85
            comp_cost_factor = 1.0 if complexity == "Low" else (1.09 if complexity == "Medium" else 1.24)
            contractor_cost_factor = 1.08 - 0.0065 * min(contractor_exp, 20)
            
            cost_mult = cost_dur_factor * inflation_factor * comp_cost_factor * contractor_cost_factor
            actual_cost = cost * cost_mult
            
            cost_noise = random.gauss(0, actual_cost * 0.065)
            actual_cost += cost_noise
            actual_cost = max(cost * 0.85, actual_cost)
            
            # Formats
            cost = round(cost, 3)
            dist_hq = round(dist_hq, 2)
            funding_rate = round(funding_rate, 2)
            inflation = round(inflation, 2)
            labor_index = round(labor_index, 2)
            actual_cost = round(actual_cost, 3)
            
            writer.writerow([
                project_id, sector, dist, complexity, scale,
                est_duration, cost, contractor_exp,
                dist_hq, funding_rate, monitoring,
                year, inflation, labor_index,
                actual_cost, actual_duration
            ])
    print(f"Finished generating {filename}.")

if __name__ == "__main__":
    # Create the data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Generate 100,000 records for each dataset to provide a "huge amount of data"
    # as requested by the user
    generate_development_dataset(os.path.join("data", "kp_subdistrict_development_index.csv"), 100000)
    generate_agricultural_dataset(os.path.join("data", "kp_agricultural_yields.csv"), 100000)
    generate_infrastructure_dataset(os.path.join("data", "kp_infrastructure_projects.csv"), 100000)
    print("All datasets generated successfully in the 'data' directory!")
