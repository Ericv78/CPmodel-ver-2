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

#-----------------------------------------------
# Input Data
#-----------------------------------------------
# All node locations (depot + customers + rendezvous points)
V = [
    (0, 0),    # 0: depot
    (15, 3),   # 1: customer node 1
    (18, 7),   # 2: customer node 2
    (4, 2),    # 3: customer node 3
    (5, 4),    # 4: customer node 4
    (8, 6),    # 5: customer node 5
    (11, 1),   # 6: rendezvous point 1
    (12, 2),   # 7: rendezvous point 2
    (10, 3),   # 8: rendezvous point 3
]

# Node type definitions (by index)
VR_input = {6, 7, 8}  # Rendezvous-only nodes (truck can visit for drone operations only)

#-----------------------------------------------
# Demands (weights)
#-----------------------------------------------
w = [
    0,   # 0: depot
    3,   # 1: customer node 1
    2,   # 2: customer node 2
    1,   # 3: customer node 3
    2,   # 4: customer node 4
    3,   # 5: customer node 5
    0,   # 6: rendezvous point 1 (no weight)
    0,   # 7: rendezvous point 2 (no weight)
    0,   # 8: rendezvous point 3 (no weight)
]

#-----------------------------------------------
# Deadlines
#-----------------------------------------------
D = {
    1: 100,  
    2: 110,  
    3: 35,   
    4: 45,
    5: 60,
}

#-----------------------------------------------
# Parameters
#-----------------------------------------------
T = 100          # Planning horizon (minutes)
E = 20           # Maximum drone endurance (battery life)
N = 1           # Number of truck-drone tandems (start with 1, scalable)
depot = 0        # Depot location

WT_max = 6     # Truck capacity
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
C = set(range(1, 6))          # Customer nodes (indices 1-5)
VL = {depot}.union(VR_input)  # Launch nodes (only depot and rendezvous points)
VR_rendezvous = VR_input      # Rendezvous node indices (from input)
VR = VR_rendezvous            # Rendezvous nodes (use the rendezvous points)
VT = set()                    # Truck-accessible customer nodes (empty - no direct truck deliveries)
VD = C                        # All customer nodes served by drone only

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
# Solution Printer
#-----------------------------------------------
def print_solution(status):
    print("\n" + "=" * 60)
    print("SOLUTION")
    print("=" * 60)
    
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print(f"Status: {solver.StatusName(status)}")
        print("No solution found.")
        return
    
    print(f"Status: {solver.StatusName(status)}")
    print(f"Total Objective: ${solver.ObjectiveValue():.2f}")
    
    # Calculate objective breakdown
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
    
    print(f"\n--- Objective Breakdown ---")
    print(f"  Truck travel cost:  ${truck_cost_val:.2f}")
    print(f"  Drone travel cost:  ${drone_cost_val:.2f}")
    print(f"  Delay penalty:      ${delay_penalty_val:.2f}")
    print(f"  Unserved penalty:   ${unserved_penalty_val:.2f}")
    print(f"  {'─' * 40}")
    print(f"  Total:              ${solver.ObjectiveValue():.2f}")
    
    print(f"\n--- Node Service Status ---")
    for i in sorted(C):
        truck_served = sum(solver.Value(var) for var in truck_service_terms[i])
        drone_served = sum(solver.Value(var) for var in drone_service_terms[i])
        served = truck_served + drone_served
        
        status_str = "✓ SERVED  " if served > 0 else "✗ UNSERVED"
        service_type = "truck" if truck_served > 0 else ("drone" if drone_served > 0 else "none")
        accessible = "truck-accessible" if i in VT else "drone-only"
        delay_val = solver.Value(delay[0, i]) if (0, i) in delay else 0
        
        print(f"  Node {i}: {status_str} | {service_type:5s} | weight: {w[i]:2d} | "
              f"delay: {delay_val:3d}min | {accessible}")
    
    print(f"\n--- Drone Flights ---")
    drone_flights = [(k, i, j, l) for (k, i, j, l) in y_drone.keys()
                     if solver.Value(y_drone[k, i, j, l]) == 1]
    
    if drone_flights:
        for k, i, j, l in drone_flights:
            launch_t = solver.Value(a[k, i])
            service_t = solver.Value(a_prime[k, j])
            rend_t = solver.Value(a[k, l])
            flight_time = t_prime[i][j] + t_prime[j][l]
            print(f"  Tandem {k}: Launch={i} (t={launch_t}) → Serve={j} (t={service_t}) → "
                  f"Rendezvous={l} (t={rend_t}) | Flight time: {flight_time}min")
    else:
        print("  None")
    
    print(f"\n--- Truck Arcs Used ---")
    truck_arcs_found = False
    for k in K:
        arcs = [(i, j) for i in range(num_nodes) for j in range(num_nodes)
                if i != j and (k, i, j) in x and solver.Value(x[k, i, j]) == 1]
        if arcs:
            truck_arcs_found = True
            print(f"  Tandem {k}: {arcs}")
    if not truck_arcs_found:
        print("  None - No truck routing arcs active!")
    
    print("=" * 60)

print_solution(status)

