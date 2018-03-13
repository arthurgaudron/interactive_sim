import numpy as np
import matplotlib.pyplot as plt
import geopy.distance
from mpl_toolkits.basemap import Basemap
from sklearn.cluster import KMeans
import vrp_solver_2opt
import geopy.distance
import numpy as np
import json
import CVRP_solver
from prettytable import PrettyTable




############################
## Traditional deliveries ##
############################

class Van_trad(object):
    def __init__(self, index, itinerary):
        self.index = index
        self.itinerary = itinerary
        self.nb_deliveries = len(itinerary) - 2
        self.total_distance = None
        self.total_duration = None
        self.loading_duration = 60
        self.handling_duration = 5
        self.average_speed = 30

    def __str__(self):
        txt = "ID: " + str(self.index) + "\n"
        txt += "Number of deliveries: " + str(self.nb_deliveries) + "\n"
        txt += "Distance: " + str(self.total_distance) + " km \n"
        txt += "Duration: " + str(self.total_duration) + " min \n"

        return txt

    def optimize_route(self):
        iti = np.asarray(self.itinerary[:-1])

        self.itinerary, index = vrp_solver_2opt.solve_tsp(iti, 0.01)

    def compute_distance_and_duration(self):
        self.total_distance = 0
        for i in range(len(self.itinerary) - 1):
            self.total_distance += geopy.distance.vincenty(self.itinerary[i], self.itinerary[i + 1]).kilometers

        self.total_distance = round(self.total_distance, 1)

        self.total_duration = 0
        self.total_duration = self.loading_duration + (self.total_distance / self.average_speed * 60) + (len(
            self.itinerary) - 1) * self.handling_duration

    def compute_time_schedule(self):
        self.time_schedule = create_time_schedule(vehicle_type="van",
                                                                 starting_time=0,
                                                                 itinerary=self.itinerary,
                                                                 loading_duration=self.loading_duration,
                                                                 handling_duration=self.handling_duration,
                                                                 average_speed=self.average_speed)



def run_traditional_delivery(sites_coordinates, vehicle_capacity = 50, timeout= 5, plot_map=True):
    """
    Run the optimization following a traditional organization.


    :param sites_coordinates:
    :return:
    """

    itineraries = CVRP_solver.solve(sites_coordinates, vehicle_capacity=vehicle_capacity, timeout=timeout)

    fleet_vans = []

    for i in range(len(itineraries)):
        fleet_vans.append(Van_trad(i, itineraries[i]))
        fleet_vans[i].compute_distance_and_duration()

    for van in fleet_vans:
        van.optimize_route()
        van.compute_distance_and_duration()
        van.compute_time_schedule()

    if plot_map:
        print("\nPLOT THE MAP OF THE NEW ITINERARIES\n")
        plot_itineraries(itineraries)

    return fleet_vans


def print_results_tradional_delivery(fleet_vans, nb_of_deliveries):
    van_duration = 0
    van_distance = 0
    van_latest_delivery = 0

    for van in fleet_vans:
        if van.total_duration > van_latest_delivery:
            van_latest_delivery = van.total_duration

        van_duration += van.total_duration
        van_distance += van.total_distance

    personel_cost = round(83 * len(fleet_vans), 2)
    vehicle_cost = round(40 * len(fleet_vans), 2)
    fuel_cost = round(van_distance/100 * 7.5 * 1.5, 2)
    total_cost = personel_cost + vehicle_cost + fuel_cost
    cost_per_delivery = round(total_cost / nb_of_deliveries, 2)

    last_delivery = van_latest_delivery
    time_trafic = round(van_duration)
    distance_driven = round(van_distance)
    co2_emission = van_distance * 0.1 * 1153
    pm_emission = van_distance * 0.1 * 0.148
    nox_emission = van_distance * 0.1 * 5.03

    kpi_list = [personel_cost, vehicle_cost, fuel_cost, total_cost, last_delivery, cost_per_delivery,
                time_trafic, distance_driven, co2_emission, pm_emission, nox_emission]

    kpi = PrettyTable()
    kpi.field_names = ["KPI", ""]
    kpi.add_row(["Personel Cost", str(personel_cost) + " euros"])
    kpi.add_row(["Vehicle Cost", str(vehicle_cost) + " euros"])
    kpi.add_row(["Fuel Cost", str(fuel_cost) + " euros"])
    kpi.add_row(["Total Cost", str(total_cost) + " euros"])
    kpi.add_row(["Cost per delivery", str(cost_per_delivery) + " euros"])
    kpi.add_row(["Final delivery done after", str(last_delivery) + " min"])
    kpi.add_row(["Time in trafic", str(time_trafic) + " min"])
    kpi.add_row(["Distance driven by van(s)", str(distance_driven) + " km"])
    kpi.add_row(["CO2 emissions", str(co2_emission) + " g"])
    kpi.add_row(["PMc emissions", str(pm_emission) + " g"])
    kpi.add_row(["NOx emissions", str(nox_emission) + " g"])

    print(kpi)

    return kpi_list


############################
##### MMDVRP deliveries ####
############################

class Van(object):
    def __init__(self):
        self.itinerary = None
        self.loading_duration = 120
        self.handling_duration = 20
        self.average_speed = 30
        self.time_schedule = None

    def generate_itinerary(self, spots_location):
        # Add the DC
        van_spots_location = np.concatenate(([[2.287224, 48.851640]], spots_location))

        self.itinerary, route_index = vrp_solver_2opt.solve_tsp(van_spots_location, improvement_threshold=0.01)


class Bike(object):
    def __init__(self, itinerary, starting_time):
        self.itinerary = itinerary
        self.total_distance = 0
        self.total_duration = 0
        self.loading_duration = 0
        self.handling_duration = 0
        self.average_speed = 20
        self.starting_time = starting_time
        self.time_schedule = None
        self.cost = 0

    def optimize_route(self):
        itinerary = np.asarray(self.itinerary)

        self.itinerary, index = vrp_solver_2opt.solve_tsp_non_circular(itinerary, 0.001)

    def compute_distance(self):
        self.total_distance = 0
        for i in range(len(self.itinerary) - 1):
            self.total_distance += geopy.distance.vincenty(self.itinerary[i], self.itinerary[i + 1]).kilometers

    def compute_duration(self):
        """
        Compute the total duration : loading + traveling time + handling * nb of deliveries
        """

        self.total_duration = self.loading_duration + \
                              (self.total_distance / self.average_speed * 60) + \
                              (len(self.itinerary) - 1) * self.handling_duration

    def compute_cost(self):
        # Deliveries decrease the cost
        unit_cost = -0.5 * (len(self.itinerary) - 1) + 5

        # Multiply per nb of deliveries
        self.cost = unit_cost * (len(self.itinerary) - 1)

    def compute_time_schedule(self):
        self.time_schedule = create_time_schedule(vehicle_type="bike",
                                                         starting_time=self.starting_time,
                                                         itinerary=self.itinerary,
                                                         loading_duration=self.loading_duration,
                                                         handling_duration=self.handling_duration,
                                                         average_speed=self.average_speed)


class Spot(object):
    def __init__(self, ref):
        self.ref = ref
        self.bikes = []
        self.points_to_deliver = None
        self.itineraries = None
        self.nb_of_bikes = None
        self.total_distance = 0
        self.total_duration = 0
        self.starting_time = 0
        self.total_cost = 0

    def __str__(self):

        txt = "Total distance: " + str(self.total_distance) + "\n"
        txt += "Total duration: " + str(self.total_duration) + "\n"
        txt += "Number of bikes: " + str(self.nb_of_bikes)
        return txt

    def update_itineraries_after_bike_optimization(self):

        self.itineraries = []

        for bike in self.bikes:
            self.itineraries.append(bike.itinerary)

    def compute_distance_and_duration(self):

        self.total_distance = 0
        self.total_duration = 0
        self.total_cost = 0
        for i in range(len(self.bikes)):
            self.total_distance += self.bikes[i].total_distance
            self.total_duration += self.bikes[i].total_duration
            self.total_cost += self.bikes[i].cost


class Problem(object):
    def __init__(self, bikes_capacity):
        self.nb_deliveries = 200
        self.dc_lng = 2.287224
        self.dc_lat = 48.851640
        self.nb_spots = 5
        self.spots = []
        self.van = None
        self.sites_coordinates = None
        self.spots_location = None
        self.points_allocation = None

        self.bikes_capacity = bikes_capacity

        self.van_distance = 0
        self.van_duration = 0

        self.bikes_distance = 0
        self.bikes_duration = 0
        self.bikes_last_delivery = 0
        self.bikes_number = 0
        self.bikes_cost = 0

    def __str__(self):

        txt = "Configuration \n"
        txt += "------------------------\n"
        txt += "Number of spots: " + str(self.nb_spots) + "\n"
        txt += "Number of deliveries: " + str(self.nb_deliveries) + "\n"
        txt += "Bike capacity: " + str(self.bikes_capacity) + "\n" + "\n"

        txt += "Van \n"
        txt += "------------------------\n"
        txt += "Van traveled distance: " + str(self.van_distance) + "\n"
        txt += "Van traveled duration: " + str(self.van_duration) + "\n" + "\n"
        txt += "Bikes \n"
        txt += "------------------------\n"
        txt += "Bikes traveled distance: " + str(self.bikes_distance) + "\n"
        txt += "Bikes traveled duration: " + str(self.bikes_duration) + "\n"
        txt += "Last bikes finishes at " + str(self.bikes_last_delivery) + "\n"
        txt += "Number of bikes: " + str(self.bikes_number)

        return txt

    def create_spots(self):
        # Create the spots
        for i in range(self.nb_spots):
            self.spots.append(Spot(i))

    def create_van(self):
        self.van = Van()

    def generate_deliveries(self):
        self.sites_coordinates = generate_deliveries(nb_of_deliveries=self.nb_deliveries,
                                                                    dc_lng=self.dc_lng,
                                                                    dc_lat=self.dc_lat)

    def compute_best_spots(self):
        self.spots_location, self.points_allocation = generate_meeting_spots(
            sites_coordinates=self.sites_coordinates,
            nb_spots=self.nb_spots)

    def compute_shortest_itineraries_spots(self):
        for spot in self.spots:

            # Select the relevant points
            spot.points_to_deliver = self.points_allocation[self.points_allocation[:, 2] == spot.ref]

            # Add the initial spot at index 0
            spot.points_to_deliver = np.vstack((self.spots_location[spot.ref, :], spot.points_to_deliver[:, :2]))

            # Transform to list
            spot.points_to_deliver = np.ndarray.tolist(spot.points_to_deliver)

            # Compute the itineraries
            spot.itineraries = CVRP_solver.solve(sites_coordinates=spot.points_to_deliver,
                                                 vehicle_capacity=self.bikes_capacity,
                                                 timeout=5)

            # Remove the last leg going back
            for i in range(len(spot.itineraries)):
                spot.itineraries[i].pop()

            spot.nb_of_bikes = len(spot.itineraries)

    def update_spots_starting_time(self):
        # Update the starting time of the spots
        for i in range(1, len(self.van.time_schedule) - 1):
            for spot in self.spots:

                # Very weak test !!!

                if self.van.time_schedule[i, 2] == spot.points_to_deliver[0][0]:
                    # Add previous waiting time + travel time
                    spot.starting_time = self.van.time_schedule[:i, 0].sum() + self.van.time_schedule[:i + 1, 1].sum()

                    break

    def create_bikes_and_allocate_itineraries(self):

        # Create the bikes for each spot
        for spot in self.spots:
            for b in range(spot.nb_of_bikes):
                spot.bikes.append(Bike(spot.itineraries[b], spot.starting_time))

        # Compute the distance of all the bikes in all the spots
        for spot in self.spots:
            for bike in spot.bikes:
                bike.optimize_route()
                bike.compute_distance()
                bike.compute_duration()
                bike.compute_cost()
                bike.compute_time_schedule()
            spot.update_itineraries_after_bike_optimization()
            spot.compute_distance_and_duration()

            self.bikes_cost += spot.total_cost

    def outputs_results(self):
        self.van_distance = self.van.time_schedule[:, 4].sum()
        self.van_duration = self.van.time_schedule[:, :2].sum()

        duration = []

        for spot in self.spots:

            self.bikes_distance += spot.total_distance

            self.bikes_number += spot.nb_of_bikes
            duration.append(spot.total_duration)

            for bike in spot.bikes:
                if self.bikes_last_delivery < bike.time_schedule[:, :2].sum():
                    self.bikes_last_delivery = bike.time_schedule[:, :2].sum()

        # Not real duration, it should remove the waiting time
        self.bikes_duration = sum(duration)


def run_MMDVRP(nb_deliveries, bikes_capacity):
    paris = Problem(bikes_capacity)
    paris.nb_deliveries = nb_deliveries
    paris.create_spots()
    paris.create_van()
    paris.generate_deliveries()
    paris.compute_best_spots()
    paris.van.generate_itinerary(paris.spots_location)
    paris.van.time_schedule = create_time_schedule(vehicle_type="van",
                                                  starting_time=0,
                                                  itinerary=paris.van.itinerary,
                                                  loading_duration=paris.van.loading_duration,
                                                  handling_duration=paris.van.handling_duration,
                                                  average_speed=paris.van.average_speed)
    paris.compute_shortest_itineraries_spots()
    paris.update_spots_starting_time()
    paris.create_bikes_and_allocate_itineraries()
    paris.outputs_results()

    personel_cost = round(83, 2)
    vehicle_cost = round(50, 2)
    fuel_cost = round(paris.van_distance/100 * 25 * 1.5, 2)
    couriers_cost = paris.bikes_cost
    total_cost = personel_cost + vehicle_cost + fuel_cost + couriers_cost
    cost_per_delivery = round(total_cost / paris.nb_deliveries, 2)

    last_delivery = paris.bikes_last_delivery
    time_trafic = round(paris.van_duration)
    distance_driven = round(paris.van_distance)
    co2_emission = paris.van_distance * 0.1 * 1153
    pm_emission = paris.van_distance * 0.1 * 0.148
    nox_emission = paris.van_distance * 0.1 * 5.03

    kpi_list = [personel_cost, vehicle_cost, fuel_cost, total_cost, last_delivery, couriers_cost, cost_per_delivery,
                time_trafic, distance_driven, co2_emission, pm_emission, nox_emission]

    kpi = PrettyTable()
    kpi.field_names = ["KPI", ""]

    kpi.add_row(["Personel Cost", str(personel_cost) + " euros"])
    kpi.add_row(["Vehicle Cost", str(vehicle_cost) + " euros"])
    kpi.add_row(["Fuel Cost", str(fuel_cost) + " euros"])
    kpi.add_row(["Couriers Cost", str(couriers_cost) + " euros"])
    kpi.add_row(["Total Cost", str(total_cost) + " euros"])
    kpi.add_row(["Cost per delivery", str(cost_per_delivery) + " euros"])
    kpi.add_row(["Final delivery done after", str(last_delivery) + " min"])
    kpi.add_row(["Time in trafic", str(time_trafic) + " min"])
    kpi.add_row(["Distance driven by van(s)", str(distance_driven) + " km"])
    kpi.add_row(["CO2 emissions", str(co2_emission) + " g"])
    kpi.add_row(["PMc emissions", str(pm_emission) + " g"])
    kpi.add_row(["NOx emissions", str(nox_emission) + " g"])

    print(kpi)

    return paris


############################
##### Common functions  ####
############################



def generate_deliveries(nb_of_deliveries, dc_lng, dc_lat):
    """

    Parameter
    ---------
    int : number of deliveries
    list : lat and lng of the DC


    Return
    ------
    list : list of the delivery points with the DC at index 0

    """

    potential_points = np.genfromtxt('potential_points.csv', delimiter=';')
    # Keeps only the coordinates
    potential_points = potential_points[:, -2:]


    index_points_to_deliver = np.random.choice(len(potential_points), nb_of_deliveries)

    # Sample the data
    # Add the DC location
    points_to_deliver = [[dc_lng, dc_lat]]

    for i in range(len(index_points_to_deliver)):
        points_to_deliver.append([potential_points[i][1], potential_points[i][0]])


    return points_to_deliver


def plot_itineraries(itineraries):
    llcrnrlon = 2.2489
    llcrnrlat = 48.8144
    urcrnrlon = 2.4211
    urcrnrlat = 48.9039

    fig = plt.figure(figsize=(10, 10))

    m = Basemap(llcrnrlon, llcrnrlat,
                urcrnrlon, urcrnrlat,
                projection='merc')

    im = plt.imread('data/map_paris.png')
    m.imshow(im, interpolation='lanczos', origin='upper')

    for i in range(len(itineraries)):
        if len(itineraries[i]) > 2:
            iti = np.asarray(itineraries[i])
            m.plot(iti[:, 0], iti[:, 1], alpha=0.7, latlon=True)


def plot_spots(spots_location, points_allocation):
    llcrnrlon = 2.2489
    llcrnrlat = 48.8144
    urcrnrlon = 2.4211
    urcrnrlat = 48.9039

    fig = plt.figure(figsize=(10, 10))

    m = Basemap(llcrnrlon, llcrnrlat,
                urcrnrlon, urcrnrlat,
                projection='merc')

    im = plt.imread('data/map_paris.png')
    m.imshow(im, interpolation='lanczos', origin='upper')

    m.scatter(points_allocation[:, 0], points_allocation[:, 1], c=points_allocation[:, 2],
                s=50, alpha=0.5, latlon=True)

    m.scatter(spots_location[:, 0], spots_location[:, 1], s=50, marker='d', latlon=True)





def compute_distance_itineraries(itineraries):
    """

    :param itineraries:
    :return:
    a list of the distance of each vehicle

    """

    vehicles_distances = []


    for i in range(len(itineraries)):
        distance = 0
        for j in range(len(itineraries[i]) - 1):
            distance += geopy.distance.vincenty(itineraries[i][j], itineraries[i][j + 1]).kilometers

        vehicles_distances.append(distance)

    return vehicles_distances



def generate_meeting_spots(sites_coordinates, nb_spots):
    # Remove the DC
    sites_coordinates = sites_coordinates[1:]
    # Convert to numpy array
    sites_coordinates = np.asarray(sites_coordinates)

    X = sites_coordinates

    kmeans = KMeans(n_clusters=nb_spots)
    kmeans.fit(X)
    results = kmeans.predict(X)

    spots_location = kmeans.cluster_centers_

    points_allocation = np.column_stack((sites_coordinates, results))

    return spots_location, points_allocation


def create_time_schedule_old(vehicle_type, starting_time, itinerary, loading_duration, handling_duration, average_speed):
    time_schedule = np.zeros(shape=(1, 4))


    for i in range(len(itinerary)):
        time_schedule = np.vstack((time_schedule, np.asarray([0, itinerary[i][0], itinerary[i][1], 0])))
        time_schedule = np.vstack((time_schedule, np.asarray([0, itinerary[i][0], itinerary[i][1], 0])))

    # Remove first point (null vector)
    time_schedule = time_schedule[1:]

    for i in range(len(time_schedule)):
        # Add loading time
        if i == 1:
            time_schedule[i, 0] = loading_duration

        # Add delivering time
        if i % 2 != 0 and i != 1:
            time_schedule[i, 0] += time_schedule[i - 1, 0]
            time_schedule[i, 0] += handling_duration
            time_schedule[i, 3] += time_schedule[i - 1, 3]

        # Add traveling time
        if i % 2 == 0 and i != 0:
            distance = geopy.distance.vincenty([time_schedule[i - 1][1], time_schedule[i - 1][2]],
                                               [time_schedule[i][1], time_schedule[i][2]]).kilometers

            travel_time = distance / average_speed * 60

            time_schedule[i, 0] += time_schedule[i - 1, 0]
            time_schedule[i, 0] += travel_time

            time_schedule[i, 3] += time_schedule[i-1, 3]
            time_schedule[i, 3] += distance

        time_schedule[i, 0] = round(time_schedule[i, 0], 3)

    if vehicle_type == "van":
        # Remove last point as there is no delivery time
        time_schedule = time_schedule[:-1]

    if vehicle_type == "bike":
        # Start the bike tours when spot delivery is over
        time_schedule[:, 0] += starting_time

    return time_schedule



def create_time_schedule(vehicle_type, starting_time, itinerary, loading_duration, handling_duration, average_speed):

    time_schedule = np.zeros(shape=(1, 5))

    for i in range(len(itinerary)):
        time_schedule = np.vstack((time_schedule, np.asarray([0, 0, itinerary[i][0], itinerary[i][1], 0])))

    # column 0 : waiting time
    # column 1: travel time to the i point from i-1 point
    # column 4 : traveled distance


    # Remove first point (null vector)
    time_schedule = time_schedule[1:]

    # Add loading time
    time_schedule[0, 0] = loading_duration

    for i in range(1, len(time_schedule)):
        time_schedule[i, 0] = handling_duration

        distance = geopy.distance.vincenty([time_schedule[i - 1][2], time_schedule[i - 1][3]],
                                           [time_schedule[i][2], time_schedule[i][3]]).kilometers

        time_schedule[i, 4] = round(distance, 2)

        travel_time = distance / average_speed * 60

        time_schedule[i, 1] = travel_time

    #if vehicle_type == "van":
        # Remove last point as there is no delivery time
        #time_schedule = time_schedule[:-1]

    if vehicle_type == "bike":
        # Start the bike tours when spot delivery is over
        time_schedule[0, 0] = starting_time + loading_duration

    return time_schedule