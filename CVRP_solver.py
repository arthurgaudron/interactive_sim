# Arthur Gaudron
# CVRP using Gurobi

from gurobipy import *
from scipy import spatial

def solve(sites_coordinates, vehicle_capacity=3, timeout=3):
    """
    Parameters
    ----------

    list : location of the DC (@index 0) and the delivery points
    int : capacity of the vehicles
    int : number of seconds allocated for the optimization

    Return
    ------

    list : list of list containing points' coordinates for each vehicle

    """

    # Define a name for each point
    sites_names = list(range(len(sites_coordinates)))
    list_of_sites = range(len(sites_names))

    # Set the demand of each site
    sites_demand = [1] * len(sites_names)

    # Compute the distance matrix
    distance_matrix = spatial.distance_matrix(sites_coordinates, sites_coordinates)

    # Remove the DC of the optimization process
    clients = list_of_sites[1:]

    model = Model('CVRP')

    x = {}
    for i in list_of_sites:
        for j in list_of_sites:
            x[i, j] = model.addVar(vtype=GRB.BINARY)

    u = {}
    for i in clients:
        u[i] = model.addVar(lb=sites_demand[i], ub=vehicle_capacity)

    model.update()

    obj = quicksum(distance_matrix[i][j] * x[i, j] for i in list_of_sites for j in list_of_sites if i != j)
    model.setObjective(obj)

    for j in clients:
        model.addConstr(quicksum(x[i, j] for i in list_of_sites if i != j) == 1)

    for i in clients:
        model.addConstr(quicksum(x[i, j] for j in list_of_sites if i != j) == 1)

    for i in clients:
        model.addConstr(u[i] <= vehicle_capacity + (sites_demand[i] - vehicle_capacity) * x[0, i])

    for i in clients:
        for j in clients:
            if i != j:
                model.addConstr(
                    u[i] - u[j] + vehicle_capacity * x[i, j] <= vehicle_capacity - sites_demand[j])

    model.setParam('TimeLimit', timeout)
    model.optimize()

    route = model.getAttr("X", x)

    # Dict is modified to get a list
    segments = []
    for key in route:
        if route[key] == 1:
            segments.append(key)

    results = list(map(list, segments))
    

    # Itineraries are made from the results
    itineraries = []

    # Each element of the list itineraries is a tour of a truck
    for i in range(len(results)):
        if results[i][0] == 0:
            itineraries.append(results[i])

    # The start point are removed from the result
    for i in range(len(itineraries)):
        results.remove(itineraries[i])

    # The itineries are built
    to_be_removed = []

    # Until there is no point to visit
    while len(results) != 0:

        # Look for each tour what is the next point to visit
        for i in range(len(itineraries)):
            # Check if the tour is over
            if itineraries[i][-1] != 0:
                for j in range(len(results)):
                    # The next point to visit is added to the itinerary
                    # and remove from the results
                    if itineraries[i][-1] == results[j][0]:
                        itineraries[i].append(results[j][0])
                        itineraries[i].append(results[j][1])
                        to_be_removed.append(results[j])

        for k in range(len(to_be_removed)):
            try:
                results.remove(to_be_removed[k])
            except:
                pass

    # This function cleans the itineraries
    def f7(seq):
        seen = set()
        seen_add = seen.add
        return [x for x in seq if not (x in seen or seen_add(x))]

    for i in range(len(itineraries)):
        itineraries[i] = f7(itineraries[i])
        itineraries[i].append(itineraries[i][0])

    # Convert indexes to coordinates
    for i in range(len(itineraries)):
        for j in range(len(itineraries[i])):
            itineraries[i][j] = sites_coordinates[itineraries[i][j]]

    return itineraries
