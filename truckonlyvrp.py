"""
===============================================================================
TRUCK-ONLY VEHICLE ROUTING PROBLEM (VRP) - CP-SAT Implementation
===============================================================================

PURPOSE:
    Incremental implementation of truck routing constraints for debugging and 
    testing individual constraint effects on route optimization.

PROBLEM DESCRIPTION:
    - Multiple trucks (N tandems) serve customer nodes from a central depot
    - Each customer has a demand (weight) that must be respected
    - Trucks have limited capacity (WT_max)
    - Some nodes are truck-accessible (VT), others are road-damaged (VD)
    - Planning horizon (T) limits the time available for all operations
    - Objective: Minimize total travel cost + penalty for unserved nodes

DECISION VARIABLES:
    - x[k,i,j]: Binary, truck k travels from node i to node j
    - u[k,i]: Integer, MTZ ordering variable for subtour elimination
    - P[k,i,j]: Binary, sequencing variable for tour ordering
    - phi[k,i]: Integer, arrival time at node i for truck k

CONSTRAINTS IMPLEMENTED:
    (36) Each customer visited at most once
    (37-38) Single depot departure/return per truck
    (39) Flow conservation (in = out at each node)
    (40) Road-damaged areas blocked for trucks
    (41-44) MTZ subtour elimination + sequencing
    (45) Capacity limit per truck
    (53) Planning horizon constraint (return to depot within time T)

OBJECTIVE FUNCTION:
    Minimize: Σ(travel_cost) + Σ(unserved_penalty)
    - Travel cost = time × cost_per_minute
    - Unserved penalty = β for each unserved customer

KEY FEATURES:
    - Scalable for any number of trucks (N) and nodes
    - Manhattan distance for truck travel times
    - Time tracking with planning horizon enforcement
    - Constraint-based approach for debugging individual constraints
    - Detailed solution output with per-truck routes and statistics

===============================================================================
"""

from ortools.sat.python import cp_model
import numpy as np

#-----------------------------------------------
#Input Data
#-----------------------------------------------
V = [
    (0, 0),     # depot
    (15, 3),    # node 1 
    (18, 7),    # node 2 
    (4, 2),     # node 3 
    (5, 4),     # node 4 
    (12, 8),    # node 5
    (20, 5),    # node 6
    (8, 12),    # node 7
    (25, 15),   # node 8
    (10, 18),   # node 9
    (30, 10),   # node 10
]

#-----------------------------------------------
#Demands (weights)
#-----------------------------------------------
w = [
    0,   # depot
    6,   # node 1
    7,   # node 2
    10,  # node 3
    2,   # node 4
    5,   # node 5
    8,   # node 6
    4,   # node 7
    9,   # node 8
    3,   # node 9
    6,   # node 10
]

#-----------------------------------------------
#Parameters
#-----------------------------------------------
vt = 1.0      # Truck speed (km/min)
ct = 2        # Truck cost per minute
WT_max = 20  # Truck capacity 
N = 2         # Number of truck tandems
depot = 0     # Location of depot
beta_value = 500.0  # penalty if unserved or served after planning horizon
T = 100       # Planning horizon (minutes)

#-----------------------------------------------
#Derived Data
#-----------------------------------------------
K = range(N)
num_nodes = len(V)
C = set(range(1, num_nodes))   # Customer nodes (excluding depot)
VT = {1, 2, 3, 4, 5, 7, 9}   # Truck-accessible customer nodes (7 out of 10)
VD = C.difference(VT)  # Road-damaged areas (nodes 6, 8, 10)
beta = {i: beta_value for i in C}

#-----------------------------------------------
#Time Matrix
#-----------------------------------------------
pts = np.array(V)
truck_dist_matrix = np.abs(pts[:, None, :] - pts[None, :, :]).sum(axis=2)
t_float = truck_dist_matrix / vt
t = np.rint(t_float).astype(int)

print("=" * 60)
print("PROBLEM SETUP")
print("=" * 60)
print(f"Nodes: {num_nodes} (1 depot + {len(C)} customers)")
print(f"Truck-accessible: {sorted(VT)}")
print(f"Road-damaged: {sorted(VD)}")
print(f"Truck capacity: {WT_max}")
print(f"Truck cost: ${ct}/min, Speed: {vt} km/min")
print(f"Unserved penalty: ${beta_value} per node")
print(f"Planning horizon: {T} minutes")
print("\nTime matrix (travel times in minutes):")
print(t)
print("=" * 60)
print()

#-----------------------------------------------
#Create Model
#-----------------------------------------------
model = cp_model.CpModel()

#-----------------------------------------------
#Decision Variables
#-----------------------------------------------
x = {}
u = {}
P = {}
phi = {}  # Arrival time at each node

# Big M for time constraints
M_time = T + max(max(row) for row in t)  # Large enough to not constrain

for k in K:
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                x[k, i, j] = model.NewBoolVar(f"x_{k}_{i}_{j}")
        # Arrival time at node i for truck k
        phi[k, i] = model.NewIntVar(0, M_time, f"phi_{k}_{i}")
    
    for i in range(1, num_nodes):
        u[k, i] = model.NewIntVar(0, num_nodes - 1, f"u_{k}_{i}")
    for i in range(num_nodes):
        for j in C:
            if i != j:
                P[k, i, j] = model.NewBoolVar(f"P_{k}_{i}_{j}")

#-----------------------------------------------
#Objective Function
#-----------------------------------------------
truck_cost = sum(t[i][j] * ct * x[k, i, j]
                 for k in K
                 for i in range(num_nodes)
                 for j in range(num_nodes) if i != j)

# Truck service credit: customer i is served by truck if any arc enters it
truck_service_terms = {
    i: [x[k, j, i] for k in K for j in range(num_nodes)
        if j != i and (k, j, i) in x]
    for i in C
}

# Unserved penalty: Σ_{i∈C} β_i (1 − Σtruck)
unserved_penalty = sum(
    beta[i] * (1 - sum(truck_service_terms[i]))
    for i in C
)

model.Minimize(truck_cost + unserved_penalty)

#-----------------------------------------------
#Constraints
#-----------------------------------------------
# (36) Each customer node visited at most once by truck
for j in C:
    truck_part = sum(x[k, i, j] for k in K for i in range(num_nodes) if i != j)
    model.Add(truck_part <= 1)

# (37, 38) Depot departure and return - if truck leaves depot, it must return
for k in K:
    departures = sum(x[k, depot, j] for j in C)
    arrivals = sum(x[k, i, depot] for i in C)
    model.Add(departures <= 1)  
    model.Add(arrivals <= 1)

    model.Add(departures == arrivals)  # Force return, can toggle on or off depending on requirements

# (39) Flow conservation
for k in K:
    for j in C:
        incoming = sum(x[k, i, j] for i in range(num_nodes) if i != j)
        outgoing = sum(x[k, j, l] for l in range(num_nodes) if l != j)
        model.Add(incoming - outgoing == 0)

# (40) Trucks cannot reach road-damaged areas
for k in K:
    for i in VD:
        for j in range(num_nodes):
            if i != j and (k, i, j) in x:
                model.Add(x[k, i, j] == 0)
                model.Add(x[k, j, i] == 0)

# (45): enforces capacity limit for truck
for k in K:
    weighted_effort = []
    for j in C:
        # Sum weight of node j if truck visits it (any incoming arc to j)
        for i in range(num_nodes):
            if i != j:
                weighted_effort.append(w[j] * x[k, i, j])
    model.Add(sum(weighted_effort) <= WT_max)

# (41,42) prevent the formation of subtours for the truck by ensuring that the truck does not traverse through previously visited arcs
M = len(C)  # maximum number of customer nodes (strict Big M)
for k in K:
    for i in range(num_nodes):
        for j in range(1, num_nodes):
            if i != j and i != depot and j != depot:
                model.Add(u[k, i] - u[k, j] + 1 <= M * (1 - x[k, i, j]))

for k in K:
    for j in range(1, num_nodes):
        incoming = sum(x[k, i, j] for i in range(num_nodes) if i != j)
        model.Add(u[k, j] <= M * incoming)

# (43,44) define the sequence of truck tours to prevent a node from being visited multiple times within a single truck route
for k in K:
    for i in range(num_nodes):
        for j in C:
            if i != j and i != depot and j != depot:
                model.Add(u[k, j] - u[k, i] <= M * P[k, i, j])
                model.Add(u[k, j] - u[k, i] >= M * (P[k, i, j] - 1) + 1)

#-----------------------------------------------
# Time Tracking Constraints  
#-----------------------------------------------
# Time propagation: if truck k travels i→j, then phi[k,j] >= phi[k,i] + t[i][j]
for k in K:
    # Departure from depot: set phi for first customer node
    for j in C:
        if (k, depot, j) in x:
            model.Add(phi[k, j] >= t[depot][j] - M_time * (1 - x[k, depot, j]))
    
    # Time propagation between customer nodes
    for i in C:
        for j in range(num_nodes):
            if i != j and (k, i, j) in x:
                model.Add(phi[k, j] >= phi[k, i] + t[i][j] - M_time * (1 - x[k, i, j]))

# (53) ensures that the arrival time of the truck at the depot does not exceed the planning horizon T
# When truck returns from node i to depot, the return time must be <= T
for k in K:
    for i in C:
        if (k, i, depot) in x:
            # If truck returns from i to depot, return time = phi[k,i] + t[i][depot]
            model.Add(phi[k, i] + t[i][depot] <= T + M_time * (1 - x[k, i, depot]))

#-----------------------------------------------
#Solve
#-----------------------------------------------
solver = cp_model.CpSolver()
status = solver.Solve(model)

#-----------------------------------------------
#Solution Printer
#-----------------------------------------------
def print_truck_routes(status):
    print("\n" + "=" * 60)
    print("SOLUTION")
    print("=" * 60)
    
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print("Status:", solver.StatusName(status))
        print("No solution found.")
        return
    
    print(f"Status: {solver.StatusName(status)}")
    print(f"Total Objective: ${solver.ObjectiveValue():.2f}")
    
    # Calculate objective breakdown
    truck_cost_val = sum(t[i][j] * ct * solver.Value(x[k, i, j])
                         for k in K
                         for i in range(num_nodes)
                         for j in range(num_nodes) if i != j)
    
    unserved_penalty_val = sum(
        beta[i] * (1 - sum(solver.Value(var) for var in truck_service_terms[i]))
        for i in C
    )
    
    print(f"\n--- Objective Breakdown ---")
    print(f"  Truck travel cost:  ${truck_cost_val:.2f}")
    print(f"  Unserved penalty:   ${unserved_penalty_val:.2f}")
    print(f"  {'─' * 40}")
    print(f"  Total:              ${truck_cost_val + unserved_penalty_val:.2f}")
    
    print(f"\n--- Node Service Status ---")
    served_count = 0
    for i in sorted(C):
        served = sum(solver.Value(var) for var in truck_service_terms[i])
        status_str = "✓ SERVED  " if served > 0 else "✗ UNSERVED"
        penalty_str = f"${beta[i] * (1 - served):.0f}" if served == 0 else "$0"
        accessible = "truck-accessible" if i in VT else "road-damaged"
        served_count += (1 if served > 0 else 0)
        print(f"  Node {i}: {status_str} | weight: {w[i]:2d} | penalty: {penalty_str:>4s} | {accessible}")
    
    print(f"\n  Summary: {served_count}/{len(C)} nodes served")
    
    print(f"\n--- Truck Routes ---")
    
    for k in K:
        arcs = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j and solver.Value(x[k, i, j]) == 1:
                    arcs.append((i, j))
        
        if arcs:
            # Build route sequence starting from depot
            route = [0]
            current = 0
            visited = {0}
            
            # Follow arcs to build route
            while True:
                next_node = None
                for (i, j) in arcs:
                    if i == current and j not in visited:
                        next_node = j
                        visited.add(j)
                        break
                
                if next_node is None:
                    break
                
                route.append(next_node)
                current = next_node
            
            # Check if there's a return to depot
            for (i, j) in arcs:
                if i == current and j == 0:
                    route.append(0)
                    break
            
            route_str = " → ".join([f"Node {n}" if n > 0 else "Depot" for n in route])
            print(f"\n  Truck {k}:")
            print(f"    Route: {route_str}")
            
            # Calculate route details
            total_time = sum(t[route[i]][route[i+1]] for i in range(len(route)-1))
            total_cost = total_time * ct
            
            # Calculate weight for this route
            route_weight = sum(w[node] for node in route if node > 0)
            
            print(f"    Travel time: {total_time} min")
            print(f"    Cost: ${total_cost}")
            print(f"    Load: {route_weight} / {WT_max} ({100*route_weight/WT_max:.1f}%)")
            print(f"    Nodes served: {len([n for n in route if n > 0])}")
        else:
            print(f"\n  Truck {k}: No route (stays at depot)")
    
    print("=" * 60)

print_truck_routes(status)