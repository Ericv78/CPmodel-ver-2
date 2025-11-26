"""
===============================================================================
TRUCK-DRONE VRP WITH RENDEZVOUS - CP-SAT Implementation
===============================================================================

PURPOSE:
    Incremental implementation of truck-drone routing with rendezvous mechanics.
    Focus on drone routing constraints with truck synchronization. Truck acts
    as mobile depot - does NOT make direct customer deliveries, only supports
    drone operations through launch and rendezvous coordination.

PROBLEM DESCRIPTION:
    - Truck-drone tandems serve customer nodes from a central depot
    - Trucks act as mobile launch platforms (no direct customer deliveries)
    - Drones launch from truck, serve customer nodes, rendezvous with truck
    - Truck travels between depot and rendezvous points to coordinate drone ops
    - Each customer has demand (weight) and delivery deadline
    - Drone capacity limits (WD_max), truck carries items for drone delivery
    - Drone endurance constraint (E minutes max flight time)
    - Planning horizon limits total operation time (T minutes)
    - Trucks can revisit the depot mid-mission to reload cargo
    - Objective: Minimize travel cost + delay penalties + unserved penalties

DECISION VARIABLES:
    - x[k,i,j]: Binary, truck k travels from node i to node j
    - y_drone[k,i,j,l]: Binary, drone k launches from i, serves j, rendezvous at l
    - a[k,i]: Integer, truck arrival time at node i
    - a_prime[k,i]: Integer, drone arrival time at node i
    - delay[k,i]: Integer, delay at node i for tandem k (lateness beyond deadline)
    - P[k,i,j]: Binary, sequencing variable for tour ordering
    - truck_load[k,i]: Integer, truck load at node i

KEY CONSTRAINTS:
    (36) Each customer visited at most once
    (40) Trucks cannot reach road-damaged areas (drone-only nodes)
    (46-48) Drone launch/rendezvous uniqueness and node restrictions
    (51-52) Initialize arrival times at depot
    (55-60) Drone travel time continuity and synchronization
    (61) Drone endurance limit
    (62) Sequential drone operations (no concurrent flights per tandem)
    (63) Delay calculation for deadline penalties
    Additional: Truck capacity tracking, truck arc activation for cost

NODE STRUCTURE:
    - Depot (0): Starting point for all tandems
    - Customer nodes (1-5): Service locations requiring drone delivery
    - Rendezvous nodes (6-8): Designated points for truck-drone coordination
    - VL = {0, 6, 7, 8}: Valid launch nodes (depot + rendezvous points)
    - VR = {6, 7, 8}: Valid rendezvous nodes (excluding depot)
    - VD = {1, 2, 3, 4, 5}: Drone-accessible customers (all customers)

SCALABLE: Start with N=1 tandem, easily scale to N>=2

===============================================================================
"""

from ortools.sat.python import cp_model
import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
except ImportError:  # Matplotlib not available in every runtime
    plt = None
    Line2D = None

#-----------------------------------------------
# Input Data
#-----------------------------------------------
# All node locations (depot + customers + rendezvous points)
V = [
    (0, 0),      # 0: depot
    (5, 2),      # 1: customer
    (1, 7),      # 2: customer
    (-5, 6),     # 3: customer
    (-6, -2),    # 4: customer
    (-1, -7),    # 5: customer
    (6, -5),     # 6: customer
    (7, 3),      # 7: customer
    (-7, 1),     # 8: customer
    (4, 1),      # 9: rendezvous point 1
    (-3, 2),     # 10: rendezvous point 2
    (-2, -3),    # 11: rendezvous point 3
    (2, -3),     # 12: rendezvous point 4
    (5, -2),     # 13: rendezvous point 5
]

# Node type definitions (by index)
VR_input = {9, 10, 11, 12, 13}

#-----------------------------------------------
# Demands (weights)
#-----------------------------------------------
w = [
    0,
    3,
    2,
    1,
    2,
    3,
    2,
    4,
    2,
    0,
    0,
    0,
    0,
    0,
]

#-----------------------------------------------
# Deadlines
#-----------------------------------------------
D = {
    1: 110,
    2: 120,
    3: 95,
    4: 85,
    5: 75,
    6: 100,
    7: 115,
    8: 105,
}

#-----------------------------------------------
# Parameters
#-----------------------------------------------
T = 100          # Planning horizon (minutes)
E = 20           # Maximum drone endurance (battery life)
N = 3           # Number of truck-drone tandems (start with 1, scalable)
depot = 0        # Depot location

WT_max = 3     # Truck capacity
WD_max = 3       # Drone capacity
ct = 0.8         # Truck cost per minute ($48/hour)
cd = 1.5         # Drone cost per minute ($90/hour)
vt = 0.6         # Truck speed (36 km/h)
vd = 0.6         # Drone speed (36 km/h)

alpha_value = 8.0      # Cost per minute of delay
beta_value = 1000.0    # Penalty if unserved

#-----------------------------------------------
# Derived Data
#-----------------------------------------------
K = range(N)
num_nodes = len(V)
C = set(range(1, 9))          # Customer nodes (1-8)
VL = {depot}.union(VR_input)  # Launch nodes (only depot and rendezvous points)
VR_rendezvous = VR_input
VR = VR_rendezvous
VT = set()
VD = C
# Allow trucks to leave/return to the depot multiple times for reload cycles.
MAX_DEPOT_TRIPS = len(V)

alpha = {i: alpha_value for i in C}
beta = {i: beta_value for i in C}

#-----------------------------------------------
# Time Matrices
#-----------------------------------------------
pts = np.array(V)
# Truck travel time (Manhattan distance)
truck_dist_matrix = np.abs(pts[:, None, :] - pts[None, :, :]).sum(axis=2)
t_float = truck_dist_matrix / vt
t = np.rint(t_float).astype(int)

# Drone travel time (Euclidean distance)
euclidean_matrix = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
t_prime_float = euclidean_matrix / vd
t_prime = np.rint(t_prime_float).astype(int)

print("=" * 60)
print("TRUCK-DRONE VRP WITH RENDEZVOUS - PROBLEM SETUP")
print("=" * 60)
print(f"Total nodes: {num_nodes} (1 depot + {len(C)} customers + {len(VR_rendezvous)} rendezvous)")
print(f"Customer nodes: {sorted(C)}")
print(f"Rendezvous-only nodes: {sorted(VR_rendezvous)}")
print(f"Truck-accessible for deliveries: {sorted(VT) if VT else 'None (drone-only deliveries)'}")
print(f"Drone-only (all customers): {sorted(VD)}")
print(f"Number of tandems: {N}")
print(f"Truck capacity: {WT_max}, Drone capacity: {WD_max}")
print(f"Truck cost: ${ct}/min, Drone cost: ${cd}/min")
print(f"Drone endurance: {E} min")
print(f"Planning horizon: {T} min")
print(f"Delay penalty: ${alpha_value}/min, Unserved penalty: ${beta_value}")
print("=" * 60)
print()

#-----------------------------------------------
# Create Model
#-----------------------------------------------
model = cp_model.CpModel()

#-----------------------------------------------
# Decision Variables
#-----------------------------------------------
# Truck routing variables
x = {}
for k in K:
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                x[k, i, j] = model.NewBoolVar(f"x_{k}_{i}_{j}")

# Drone routing variables (i=launch, j=customer, l=rendezvous)
y_drone = {}
for k in K:
    for i in VL:
        for j in C:
            for l in VR:
                if i != j and j != l and i != l:
                    y_drone[k, i, j, l] = model.NewBoolVar(f"y_drone_{k}_{i}_{j}_{l}")

# Arrival time variables
a = {}        # Truck arrival times
a_prime = {}  # Drone arrival times
for k in K:
    for i in range(num_nodes):
        a[k, i] = model.NewIntVar(0, T, f"a_{k}_{i}")
        a_prime[k, i] = model.NewIntVar(0, T, f"a_prime_{k}_{i}")

# Delay variables
delay = {}
for k in K:
    for i in C:
        delay[k, i] = model.NewIntVar(0, T, f"delay_{k}_{i}")

# Sequencing variables (for constraint 62)
P = {}
for k in K:
    for i in VL:
        for j in C:
            if i != j:
                P[k, i, j] = model.NewBoolVar(f"P_{k}_{i}_{j}")

# Truck capacity tracking variables
# truck_load[k, i] = current load on truck k when arriving at node i
truck_load = {}
for k in K:
    for i in range(num_nodes):
        truck_load[k, i] = model.NewIntVar(0, WT_max, f"truck_load_{k}_{i}")

# Truck usage indicator (1 if truck k leaves the depot)
truck_active = {k: model.NewBoolVar(f"truck_active_{k}") for k in K}


#-----------------------------------------------
# Objective Function 
#-----------------------------------------------

# 1. Truck travel cost
truck_cost = sum(t[i][j] * ct * x[k, i, j]
                 for k in K
                 for i in range(num_nodes)
                 for j in range(num_nodes) if i != j)

# 2. Drone travel cost (launch to customer to rendezvous)
drone_cost = sum((t_prime[i][j] + t_prime[j][l]) * cd * y_drone[k, i, j, l]
                 for k in K
                 for i in VL
                 for j in C
                 for l in VR
                 if (k, i, j, l) in y_drone)

# 3. Delay penalty
delay_penalty = sum(alpha[i] * delay[k, i] for k in K for i in C)

# 4. Service tracking for unserved penalty
# Truck service credit: node i served if truck arrives at i
truck_service_terms = {
    i: [x[k, j, i] for k in K for j in VL
        if j != i and (k, j, i) in x]
    for i in C
}

# Drone service credit: node i served if drone delivers to i
drone_service_terms = {
    i: [y_drone[k, launch, i, rend] for k in K 
        for launch in VL for rend in VR
        if (k, launch, i, rend) in y_drone]
    for i in C
}

# 5. Unserved penalty
unserved_penalty = sum(
    beta[i] * (1 - (sum(truck_service_terms[i]) + sum(drone_service_terms[i])))
    for i in C
)

# Final objective: minimize total cost
model.Minimize(truck_cost + drone_cost + delay_penalty + unserved_penalty)

#-----------------------------------------------
# Constraints
#-----------------------------------------------

# (36) Each affected area visited at most once (truck or drone)
for j in C:
    truck_part = sum(x[k, i, j] for k in K for i in VL if i != j) #always equal 0, due to no truck deliveries
    drone_part = sum(y_drone[k, i, j, l] for k in K for i in VL if i != j 
                     for l in VR if l != i and (k, i, j, l) in y_drone)
    model.Add(truck_part + drone_part <= 1)

# (40) Trucks cannot reach road-damaged areas (all customer nodes in this case)
# This prevents trucks from visiting any customer nodes - they can only visit depot and rendezvous nodes
for k in K:
    for i in VD:
        for j in range(num_nodes):
            if i != j and (k, i, j) in x:
                model.Add(x[k, i, j] == 0)
                model.Add(x[k, j, i] == 0)

# (47, 48) The drone can be launched and returned only once per node
# (47) Each launch node can launch at most one drone per tandem
for k in K:
    for i in VL:
        launch_trips = []
        for j in C:
            if j != i:
                for l in VR:
                    if l != i and l != j and (k, i, j, l) in y_drone:
                        launch_trips.append(y_drone[k, i, j, l])
        if launch_trips:
            model.Add(sum(launch_trips) <= 1)

# (48) Each rendezvous node can receive at most one drone per tandem
for k in K:
    for l in VR:
        rendezvous_trips = []
        for i in VL:
            if i != l:
                for j in C:
                    if j != i and j != l and (k, i, j, l) in y_drone:
                        rendezvous_trips.append(y_drone[k, i, j, l])
        if rendezvous_trips:
            model.Add(sum(rendezvous_trips) <= 1)

# (51, 52) Initialize arrival times at depot to zero
for k in K:
    model.Add(a[k, 0] == 0)
    model.Add(a_prime[k, 0] == 0)

# (55, 56) Drone arrival time continuity - ensures drone arrives at nodes sequentially
# (55) Drone travel from launch node i to customer j
for k in K:
    for i in VL:
        for j in C:
            if i == j:
                continue
            # Collect all possible flights (i,j,l)
            flights_ijl = [y_drone[k, i, j, l]
                           for l in VR
                           if l != i and l != j and (k, i, j, l) in y_drone]
            if flights_ijl:
                sum_ijl = sum(flights_ijl)
                model.Add(a[k, i] + t_prime[i][j] - T * (1 - sum_ijl) <= a_prime[k, j])

# (56) Drone travel from customer j to rendezvous l
for k in K:
    for j in C:
        for l in VR:
            if j == l:
                continue
            flights_ijl = [y_drone[k, i, j, l]
                           for i in VL
                           if i != j and i != l and (k, i, j, l) in y_drone]
            if flights_ijl:
                sum_ijl = sum(flights_ijl)
                model.Add(a_prime[k, j] + t_prime[j][l] - T * (1 - sum_ijl) <= a[k, l])

# (57-60) Synchronize truck and drone arrival times at launch and rendezvous nodes
# (57, 58) Launch synchronization - drone and truck must be at same location at launch
for k in K:
    for i in VL:
        terms = [
            y_drone[k, i, j, l]
            for j in C if j != i
            for l in VR if l != i and l != j and (k, i, j, l) in y_drone
        ]
        if terms:
            sortie_sum = sum(terms)
            model.Add(a_prime[k, i] >= a[k, i] - T * (1 - sortie_sum))  # (57)
            model.Add(a_prime[k, i] <= a[k, i] + T * (1 - sortie_sum))  # (58)

# (59, 60) Rendezvous synchronization - drone and truck must meet at rendezvous location
for k in K:
    for l in VR:
        terms = [
            y_drone[k, i, j, l]
            for i in VL if i != l
            for j in C if j != i and j != l and (k, i, j, l) in y_drone
        ]
        if terms:
            sortie_sum = sum(terms)
            model.Add(a_prime[k, l] >= a[k, l] - T * (1 - sortie_sum))  # (59)
            model.Add(a_prime[k, l] <= a[k, l] + T * (1 - sortie_sum))  # (60)

# (61) Drone endurance constraint - total flight time cannot exceed battery capacity
for k in K:
    for i in VL:
        for j in C:
            for l in VR:
                if i != j and i != l and j != l and (k, i, j, l) in y_drone:
                    model.Add(
                        t_prime[i][j]        # Launch to customer
                        + t_prime[j][l]      # Customer to rendezvous
                        - T * (1 - y_drone[k, i, j, l])
                        <= E
                    )

# Drone capacity constraint - prevent drones from serving nodes exceeding weight capacity
for k in K:
    for i in VL:
        for j in C:
            for l in VR:
                if i != j and j != l and i != l and (k, i, j, l) in y_drone:
                    if w[j] > WD_max:
                        model.Add(y_drone[k, i, j, l] == 0)

# (46) Drones are restricted to serving affected areas within a set (VD)
# Explicitly prevent drones from serving nodes not in VD (drone-only nodes)
for k in K:
    for i in VL:
        for j in C:
            for l in VR:
                if (
                    i == j or i == l or
                    j == i or j == l or j not in VD or
                    l == i or l == j
                ):
                    if (k, i, j, l) in y_drone:
                        model.Add(y_drone[k, i, j, l] == 0)

# Truck flow conservation and depot return
for k in K:
    # Balance flows at the depot so a truck can depart-reload multiple times.
    outgoing_from_depot = sum(
        x[k, depot, j]
        for j in range(num_nodes)
        if j != depot and (k, depot, j) in x
    )
    incoming_to_depot = sum(
        x[k, i, depot]
        for i in range(num_nodes)
        if i != depot and (k, i, depot) in x
    )
    model.Add(outgoing_from_depot == incoming_to_depot)
    model.Add(outgoing_from_depot >= truck_active[k])
    model.Add(outgoing_from_depot <= MAX_DEPOT_TRIPS * truck_active[k])

    for node in VL:
        if node == depot:
            continue
        incoming = sum(
            x[k, i, node]
            for i in range(num_nodes)
            if i != node and (k, i, node) in x
        )
        outgoing = sum(
            x[k, node, j]
            for j in range(num_nodes)
            if j != node and (k, node, j) in x
        )
        model.Add(incoming == outgoing)

    # Any drone sortie requires the truck to be active
    for i in VL:
        for j in C:
            for l in VR:
                if (k, i, j, l) in y_drone:
                    model.Add(truck_active[k] >= y_drone[k, i, j, l])


# (62) Prevents trucks from launching drones that are still delivering
# Ensures sequential operations - drone must complete before truck can launch another
for k in K:
    for i in VL:
        for l in VR:
            for b in C:
                if i != b and i != l and l != b:
                    # First sum: Σ_{j ∈ C \ {i,l}} y_{i j l}^k
                    sum1_terms = [
                        y_drone[k, i, j, l]
                        for j in C
                        if j != i and j != l and (k, i, j, l) in y_drone
                    ]

                    # Second sum: Σ_{q ∈ C \ {b,m}} Σ_{m ∈ VR \ {b,q}} y_{b q m}^k
                    sum2_terms = [
                        y_drone[k, b, q, m]
                        for q in C if q != b
                        for m in VR if m != b and m != q and (k, b, q, m) in y_drone
                    ]

                    # Only add if at least one term exists
                    if sum1_terms or sum2_terms or (k, l, b) in P:
                        sum1 = sum(sum1_terms) if sum1_terms else 0
                        sum2 = sum(sum2_terms) if sum2_terms else 0
                        P_var = P[k, l, b] if (k, l, b) in P else 0

                        model.Add(
                            a_prime[k, l]
                            - T * (3 - sum1 - sum2 - P_var)
                            <= a_prime[k, b]
                        )

#-----------------------------------------------
# TRUCK TRAVEL ARC ACTIVATION
#-----------------------------------------------
# When a drone launches from i and rendezvous at l, the truck must travel from i to l
# This ensures truck travel cost is accounted for
for k in K:
    for i in VL:
        for j in C:
            for l in VR:
                if i != j and j != l and i != l and (k, i, j, l) in y_drone:
                    # If drone flight y_drone[k,i,j,l] happens, truck must travel i→l
                    if (k, i, l) in x:
                        model.Add(x[k, i, l] >= y_drone[k, i, j, l])

# (49) The drone can be launched and retrieved at different nodes along the truck route
# Ensures truck has outgoing arcs from launch node and incoming arcs to rendezvous node
for k in K:
    for i in VL:
        for j in C:
            for l in VR:
                if i == l or i == j or j == l:
                    continue
                if (k, i, j, l) not in y_drone:
                    continue
                y_var = y_drone[k, i, j, l]
                # Truck must have outgoing arc from launch node i
                sum_out_i = sum(x[k, i, t] for t in range(num_nodes) if t != i and (k, i, t) in x)
                # Truck must have incoming arc to rendezvous node l
                sum_in_l = sum(x[k, t, l] for t in range(num_nodes) if t != l and (k, t, l) in x)
                # If drone flies, truck must depart from i and arrive at l
                model.Add(2 * y_var <= sum_out_i + sum_in_l)

# (50) Mandates that the associated truck must depart from any node to reach the rendezvous node l
# Specifically ensures truck arrives at rendezvous when drone launches from depot
for k in K:
    for j in C:
        for l in VR:
            if j != l and (k, depot, j, l) in y_drone:
                rhs = sum(x[k, i, l] for i in VL if i != j and i != l and (k, i, l) in x)
                model.Add(y_drone[k, depot, j, l] <= rhs)

#-----------------------------------------------
# TRUCK CAPACITY TRACKING CONSTRAINTS
#-----------------------------------------------
# The truck starts at depot with full capacity
for k in K:
    model.Add(truck_load[k, depot] == WT_max)

# Capacity flow: When truck travels from node i to node j
# If a drone launches from i to serve customer c (and rendezvous at l),
# the truck loses w[c] capacity
for k in K:
    for j in range(num_nodes):
        if j == depot:
            # When returning to depot, truck refills to full capacity
            model.Add(truck_load[k, depot] == WT_max)
        else:
            # For non-depot nodes, calculate capacity based on incoming arcs
            for i in range(num_nodes):
                if i != j and (k, i, j) in x:
                    # When truck travels from i to j, we need to account for drone launches from i
                    # If truck goes from i to j, capacity at j = capacity at i - sum of drone launches from i
                    
                    # Calculate total weight of drones launched from node i
                    drone_weight_from_i = sum(
                        w[c] * y_drone[k, i, c, l]
                        for c in C
                        for l in VR
                        if (k, i, c, l) in y_drone
                    )
                    
                    # Capacity constraint: if truck travels from i to j, 
                    # truck_load[k,j] = truck_load[k,i] - drone_weight_from_i
                    model.Add(
                        truck_load[k, j] == truck_load[k, i] - drone_weight_from_i
                    ).OnlyEnforceIf(x[k, i, j])

# Ensure truck has enough capacity for any drone launch
for k in K:
    for i in VL:
        for c in C:
            for l in VR:
                if (k, i, c, l) in y_drone:
                    # If drone launches from i to serve c, truck must have at least w[c] capacity at i
                    model.Add(truck_load[k, i] >= w[c]).OnlyEnforceIf(y_drone[k, i, c, l])

#-----------------------------------------------
# Solve
#-----------------------------------------------
solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 30
solver.parameters.num_search_workers = 8
status = solver.Solve(model)

#-----------------------------------------------
# Solution extraction and reporting helpers
#-----------------------------------------------
def collect_solution_data(status_code):
    data = {
        "status_code": status_code,
        "status_name": solver.StatusName(status_code),
    }

    if status_code not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return data

    truck_cost_val = sum(t[i][j] * ct * solver.Value(x[k, i, j])
                         for k in K
                         for i in range(num_nodes)
                         for j in range(num_nodes) if i != j)

    drone_cost_val = sum((t_prime[i][j] + t_prime[j][l]) * cd * solver.Value(y_drone[k, i, j, l])
                         for k in K
                         for i in VL for j in C for l in VR
                         if (k, i, j, l) in y_drone)

    delay_penalty_val = sum(alpha[i] * solver.Value(delay[k, i]) for k in K for i in C)

    unserved_penalty_val = sum(
        beta[i] * (1 - (sum(solver.Value(var) for var in truck_service_terms[i]) +
                        sum(solver.Value(var) for var in drone_service_terms[i])))
        for i in C
    )

    node_service = {}
    for i in sorted(C):
        truck_served = sum(solver.Value(var) for var in truck_service_terms[i])
        drone_served = sum(solver.Value(var) for var in drone_service_terms[i])
        delay_val = solver.Value(delay[0, i]) if (0, i) in delay else 0
        node_service[i] = {
            "truck_served": truck_served,
            "drone_served": drone_served,
            "delay": delay_val,
            "accessible": "truck-accessible" if i in VT else "drone-only",
        }

    drone_flights = []
    for (k, i, j, l), var in y_drone.items():
        if solver.Value(var) == 1:
            drone_flights.append({
                "tandem": k,
                "launch": i,
                "customer": j,
                "rendezvous": l,
                "launch_time": solver.Value(a[k, i]),
                "service_time": solver.Value(a_prime[k, j]),
                "rendezvous_time": solver.Value(a[k, l]),
                "flight_time": t_prime[i][j] + t_prime[j][l],
            })

    truck_arcs = {}
    for k in K:
        arcs = [(i, j) for i in range(num_nodes) for j in range(num_nodes)
                if i != j and (k, i, j) in x and solver.Value(x[k, i, j]) == 1]
        if arcs:
            truck_arcs[k] = arcs

    depot_departures = {
        k: sum(
            solver.Value(x[k, depot, j])
            for j in range(num_nodes)
            if j != depot and (k, depot, j) in x
        )
        for k in K
    }
    cargo_recharges = {k: max(0, departures - 1) for k, departures in depot_departures.items()}

    truck_arrivals = {
        k: {i: solver.Value(a[k, i]) for i in range(num_nodes)}
        for k in K
    }

    drone_arrivals = {
        k: {i: solver.Value(a_prime[k, i]) for i in range(num_nodes)}
        for k in K
    }

    data.update({
        "objective": solver.ObjectiveValue(),
        "truck_cost": truck_cost_val,
        "drone_cost": drone_cost_val,
        "delay_penalty": delay_penalty_val,
        "unserved_penalty": unserved_penalty_val,
        "node_service": node_service,
        "drone_flights": drone_flights,
        "truck_arcs": truck_arcs,
        "truck_arrivals": truck_arrivals,
        "drone_arrivals": drone_arrivals,
        "truck_active": {k: solver.Value(truck_active[k]) for k in K},
        "cargo_recharges": cargo_recharges,
        "depot_departures": depot_departures,
    })

    return data


def print_solution(solution_data):
    print("\n" + "=" * 60)
    print("SOLUTION")
    print("=" * 60)

    status_code = solution_data["status_code"]
    print(f"Status: {solution_data['status_name']}")

    if status_code not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print("No solution found.")
        print("=" * 60)
        return

    print(f"Total Objective: ${solution_data['objective']:.2f}")

    print(f"\n--- Objective Breakdown ---")
    print(f"  Truck travel cost:  ${solution_data['truck_cost']:.2f}")
    print(f"  Drone travel cost:  ${solution_data['drone_cost']:.2f}")
    print(f"  Delay penalty:      ${solution_data['delay_penalty']:.2f}")
    print(f"  Unserved penalty:   ${solution_data['unserved_penalty']:.2f}")
    print(f"  {'─' * 40}")
    print(f"  Total:              ${solution_data['objective']:.2f}")

    print(f"\n--- Node Service Status ---")
    for i in sorted(solution_data["node_service"].keys()):
        node_info = solution_data["node_service"][i]
        served = node_info["truck_served"] + node_info["drone_served"]
        status_str = "✓ SERVED  " if served > 0 else "✗ UNSERVED"
        if node_info["truck_served"]:
            service_type = "truck"
        elif node_info["drone_served"]:
            service_type = "drone"
        else:
            service_type = "none"
        print(f"  Node {i}: {status_str} | {service_type:5s} | weight: {w[i]:2d} | "
              f"delay: {node_info['delay']:3d}min | {node_info['accessible']}")

    print(f"\n--- Drone Flights ---")
    if solution_data["drone_flights"]:
        for flight in solution_data["drone_flights"]:
            print(
                f"  Tandem {flight['tandem']}: Launch={flight['launch']} (t={flight['launch_time']}) "
                f"→ Serve={flight['customer']} (t={flight['service_time']}) "
                f"→ Rendezvous={flight['rendezvous']} (t={flight['rendezvous_time']}) "
                f"| Flight time: {flight['flight_time']}min"
            )
    else:
        print("  None")

    print(f"\n--- Truck Arcs Used ---")
    if solution_data["truck_arcs"]:
        for k, arcs in solution_data["truck_arcs"].items():
            print(f"  Tandem {k}: {arcs}")
    else:
        print("  None - No truck routing arcs active!")

    recharges = solution_data.get("cargo_recharges", {})
    print(f"\n--- Depot Cargo Recharges ---")
    if recharges:
        for k in sorted(recharges.keys()):
            label = "recharges" if recharges[k] != 1 else "recharge"
            print(f"  Tandem {k}: {recharges[k]} {label}")
    else:
        print("  Not tracked")

    print("=" * 60)


def plot_solution(solution_data, show=True, save_path=None):
    if plt is None:
        print("Matplotlib is not installed; skipping visualization.")
        return

    if solution_data["status_code"] not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print("No feasible solution to plot.")
        return

    def _palette_pair(index):
        cmap = plt.colormaps.get_cmap("tab20")
        colors = cmap.colors if hasattr(cmap, "colors") else cmap(np.linspace(0, 1, 20))
        truck_color = colors[(2 * index) % len(colors)]
        drone_color = colors[(2 * index + 1) % len(colors)]
        return truck_color, drone_color

    def _draw_base_nodes(ax):
        coords = np.array(V)
        ax.scatter(coords[depot, 0], coords[depot, 1], c="black", marker="s", s=120)
        rendezvous_idx = sorted(VR)
        if rendezvous_idx:
            ax.scatter(coords[rendezvous_idx, 0], coords[rendezvous_idx, 1], c="black", marker="^", s=95)

        served_idx = []
        unserved_idx = []
        for i in sorted(C):
            node_info = solution_data["node_service"].get(i, {})
            served = node_info.get("truck_served", 0) + node_info.get("drone_served", 0)
            if served:
                served_idx.append(i)
            else:
                unserved_idx.append(i)

        if served_idx:
            served_coords = coords[served_idx]
            ax.scatter(served_coords[:, 0], served_coords[:, 1], c="tab:green", marker="o", s=75)
        if unserved_idx:
            unserved_coords = coords[unserved_idx]
            ax.scatter(unserved_coords[:, 0], unserved_coords[:, 1], c="tab:red", marker="o", s=75)

        for idx, (x_coord, y_coord) in enumerate(V):
            ax.text(x_coord + 0.2, y_coord + 0.2, str(idx), fontsize=8)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle=":", linewidth=0.5)
        ax.set_xlabel("X coordinate")
        ax.set_ylabel("Y coordinate")

    num_trucks = max(N, 1)
    fig = plt.figure(figsize=(max(12, 4 * num_trucks), 8))
    gs = fig.add_gridspec(2, num_trucks, height_ratios=[1, 1.1], hspace=0.3, wspace=0.25)

    truck_axes = []
    for col in range(num_trucks):
        ax = fig.add_subplot(gs[0, col])
        _draw_base_nodes(ax)
        ax.set_title(f"Truck {col} arcs")
        truck_axes.append(ax)

    ax_drone = fig.add_subplot(gs[1, :])
    _draw_base_nodes(ax_drone)
    ax_drone.set_title("Drone arcs (all tandems)")

    legend_handles = [
        Line2D([], [], marker="s", color="none", markerfacecolor="black", markersize=8, label="Depot"),
        Line2D([], [], marker="^", color="none", markerfacecolor="black", markersize=8, label="Rendezvous"),
        Line2D([], [], marker="o", color="none", markerfacecolor="tab:green", markersize=8, label="Served customer"),
        Line2D([], [], marker="o", color="none", markerfacecolor="tab:red", markersize=8, label="Unserved customer"),
    ]

    x_min = min(v[0] for v in V) - 1
    x_max = max(v[0] for v in V) + 1
    y_min = min(v[1] for v in V) - 1
    y_max = max(v[1] for v in V) + 1

    for ax in truck_axes:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
    ax_drone.set_xlim(x_min, x_max)
    ax_drone.set_ylim(y_min, y_max)

    for k in range(N):
        truck_color, drone_color = _palette_pair(k)
        arcs = solution_data["truck_arcs"].get(k, [])
        ax_truck_k = truck_axes[k]
        if arcs:
            for (i, j) in arcs:
                xs = [V[i][0], V[j][0]]
                ys = [V[i][1], V[j][1]]
                ax_truck_k.plot(xs, ys, color=truck_color, linewidth=2.0, alpha=0.6)
            legend_handles.append(Line2D([], [], color=truck_color, linewidth=2.0, label=f"Tandem {k}: truck"))

        flights = [f for f in solution_data["drone_flights"] if f["tandem"] == k]
        if flights:
            for flight in flights:
                launch = V[flight["launch"]]
                customer = V[flight["customer"]]
                rendezvous = V[flight["rendezvous"]]
                ax_drone.plot([launch[0], customer[0]], [launch[1], customer[1]], color=drone_color,
                              linewidth=2.2, alpha=0.7)
                ax_drone.plot([customer[0], rendezvous[0]], [customer[1], rendezvous[1]], color=drone_color,
                              linewidth=2.2, alpha=0.7)
                ax_drone.annotate("", xy=rendezvous, xytext=customer,
                                  arrowprops=dict(arrowstyle="->", color=drone_color, lw=0.9, alpha=0.6))
                ax_drone.annotate("", xy=customer, xytext=launch,
                                  arrowprops=dict(arrowstyle="->", color=drone_color, lw=0.9, alpha=0.6))
            legend_handles.append(Line2D([], [], color=drone_color, linewidth=2.2, label=f"Tandem {k}: drone"))

    fig.legend(handles=legend_handles, loc="lower left", ncol=1, fontsize=8, bbox_to_anchor=(0.02, 0.04))
    fig.suptitle("Truck and Drone Routing Map", fontweight="bold", y=0.97)

    fig.tight_layout(rect=[0.05, 0.08, 0.98, 0.96])

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main(show_plot=True, save_path=None):
    solution_data = collect_solution_data(status)
    print_solution(solution_data)
    if show_plot or save_path:
        plot_solution(solution_data, show=show_plot, save_path=save_path)
    return solution_data


if __name__ == "__main__":
    main(show_plot=True)
