import random

projects = [
    {"site_id": "MHP-0001", "cable_length_km": 12.0, "cost_per_km_million_pkr": 0.9},
    {"site_id": "MHP-0002", "cable_length_km": 30.0, "cost_per_km_million_pkr": 0.9},
    {"site_id": "MHP-0003", "cable_length_km": 5.0,  "cost_per_km_million_pkr": 1.4},
]

def rough_cable_cost(project):
    return project["cable_length_km"] * project["cost_per_km_million_pkr"]

for p in projects:
    cost=rough_cable_cost(p)
    print(f"{p['site_id']}: approx {cost:.2f} million PKR of cable cost")


def kw_to_mw(capacity_kw):
    return capacity_kw /1000

capacities_kw = [100.0, 250.0, 500.0]
#print(capacities_kw * 1000)   # not what you think it does



test=random.choices(range(1,4000),k=50)

rst=[ y*5 if y <250 else y*65 for y in test]
rst_sorted = sorted(rst)
for p in rst_sorted:
    if p ==250:
        print(f"Protected Status Bill={p} ")
    else:
        print (f"Unprotected Bill is {p} ")

