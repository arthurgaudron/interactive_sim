import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual, FloatSlider, IntSlider, Layout
import numpy as np
from prettytable import PrettyTable


style = {'description_width': 'initial'}
layout = Layout(width='60%')

def interactive_plot():
    interactive_plot = interact(f,
                                   fuel_cost=FloatSlider(description='Fuel cost (€/L): ',
                                                         value=1.1,
                                                         min=0.01,
                                                         max=2,
                                                         step=0.01,
                                                         style=style, layout=layout),

                                   driver_cost_van=FloatSlider(description='VRP - Van driver cost (€/d): ',
                                                               value=83,
                                                               min=1,
                                                               max=100,
                                                               step=1,
                                                               style=style,
                                                               layout=layout),

                                   vehicle_cost_van=FloatSlider(description='VRP - Van cost (€/v): ',
                                                                value=40,
                                                                min=1,
                                                                max=100,
                                                                step=1,
                                                                style=style,
                                                                layout=layout),

                                   fuel_consup_van=FloatSlider(description='VRP - Van fuel consumption (L/100km): ',
                                                               value=7,
                                                               min=0,
                                                               max=15,
                                                               step=0.1,
                                                               style=style,
                                                               layout=layout),


                                   pm_van=FloatSlider(description='VRP - Van PM emission (g/km): ',
                                                      value=0.5,
                                                      min=0.1,
                                                      max=1,
                                                      step=0.01,
                                                      style=style,
                                                      layout=layout),

                                   capacity_van=IntSlider(description='VRP - Van capacity (#): ',
                                                      value=80,
                                                      min=5,
                                                      max=100,
                                                      step=5,
                                                      style=style,
                                                      layout=layout),


                                   dropoff_duration_van=FloatSlider(description='VRP - Dropoff duration (min): ',
                                                                    value=5,
                                                                    min=1,
                                                                    max=30,
                                                                    style=style,
                                                                    layout=layout),

                                   speed_van=FloatSlider(description='VRP - Van speed (km/h): ',
                                                         value=30,
                                                         min=1,
                                                         max=50,
                                                         style=style,
                                                         layout=layout),

                                   CC_cost_van=FloatSlider(description='VRP - Van climate change cost (cts€/km): ',
                                                           value = 2.8,
                                                           min=0.1,
                                                           max=5,
                                                           style=style,
                                                           layout=layout),

                                   PH_cost_van=FloatSlider(description='VRP - Van public health cost (cts€/km): ',
                                                           value=3,
                                                           min=1.1,
                                                           max=5.9,
                                                           style=style,
                                                           layout=layout),

                                   driver_cost_truck=FloatSlider(description='MDVRP - Truck driver cost (€/d): ',
                                                      value=83,
                                                      min=1,
                                                      max=100,
                                                      step=1,
                                                      style=style,
                                                      layout=layout),


                                   vehicle_cost_truck=FloatSlider(description='MDVRP - Truck cost (€/t): ',
                                                      value=50,
                                                      min=1,
                                                      max=120,
                                                      step=1,
                                                      style=style,
                                                      layout=layout),

                                   courrier_cost=FloatSlider(description='MDVRP - Courrier cost (€/delivery): ',
                                                      value=5,
                                                      min=0.1,
                                                      max=10,
                                                      step=0.1,
                                                      style=style,
                                                      layout=layout),

                                   fuel_consup_truck=FloatSlider(description='MDVRP - Truck fuel consumption (L/100km): ',
                                                      value=25,
                                                      min=0,
                                                      max=40,
                                                      step=0.1,
                                                      style=style,
                                                      layout=layout),

                                   pm_truck=FloatSlider(description='MDVRP - Truck particle emission (g/tkm): ',
                                                      value=0.308,
                                                      min=0.1,
                                                      max=1,
                                                      step=0.001,
                                                      style=style,
                                                      layout=layout),

                                   capacity_bike=FloatSlider(description='MDVRP - Bike capacity (#): ',
                                                      value=5,
                                                      min=1,
                                                      max=20,
                                                      step=1,
                                                      style=style,
                                                      layout=layout),

                                   dropoff_duration_truck=FloatSlider(description='MDVRP - Truck dropoff duration (min): ',
                                                      value=15,
                                                      min=0,
                                                      max=30,
                                                      step=1,
                                                      style=style,
                                                      layout=layout),

                                   dropoff_duration_bike=FloatSlider(description='MDVRP - Bike dropoff duration (min): ',
                                                      value=5,
                                                      min=0,
                                                      max=15,
                                                      step=1,
                                                      style=style,
                                                      layout=layout),

                                   speed_truck=FloatSlider(description='MDVRP - Truck speed (km/h): ',
                                                      value=30,
                                                      min=1,
                                                      max=50,
                                                      step=1,
                                                      style=style,
                                                      layout=layout),

                                   speed_bike=FloatSlider(description='MDVRP - Bike speed (km/h): ',
                                                      value=20,
                                                      min=1,
                                                      max=50,
                                                      step=1,
                                                      style=style,
                                                      layout=layout),

                                   CC_cost_truck=FloatSlider(description='MDVRP - Truck climate change cost (cts€/km): ',
                                                             value=1.7,
                                                             min=0,
                                                             max=2.9,
                                                             step=0.1,
                                                             style=style,
                                                             layout=layout),

                                   PH_cost_truck=FloatSlider(description='MDVRP - Truck air pollution cost (cts€/km): ',
                                                                value=2.80,
                                                                min=0,
                                                                max=5.9,
                                                                step=0.1,
                                                                style=style,
                                                                layout=layout),
                                continous_update=False,

                                   )


    return interactive_plot


def f(fuel_cost, driver_cost_van, vehicle_cost_van, fuel_consup_van,
      pm_van,
      capacity_van,
      dropoff_duration_van,
      speed_van,
      CC_cost_van,
      PH_cost_van,
      driver_cost_truck,
      vehicle_cost_truck,
      courrier_cost,
      fuel_consup_truck,
      pm_truck,
      capacity_bike,
      dropoff_duration_truck,
      dropoff_duration_bike,
      speed_truck,
      speed_bike,
      CC_cost_truck,
      PH_cost_truck):
    # Shared variables
    nb_deliveries = 200

    # # VRP variables
    # driver_cost_van = 50
    # vehicle_cost_van = 30
    # fuel_consup_van = 9
    # speed_van = 30
    # pm_van = 100
    # dropoff_duration_van = 15

    # Load the data
    data_vrp = pd.read_csv("Gurobi_VRP/vrp_ewgt t_20Sec.csv", sep=";", dtype="float")
    data_mdvrp = pd.read_csv("Gurobi_MDVRP/mdvrp_ewgt_smoothen.csv", sep=";", dtype="float")

    scenario_vrp = data_vrp[data_vrp["Capacity"] == capacity_van]
    scenario_mdvrp = data_mdvrp[data_mdvrp["Capacity"] == capacity_bike]


    # VRP scenario
    total_driver_cost_vrp = scenario_vrp["Nb of vehicles"] * driver_cost_van

    total_vehicule_cost_vrp = scenario_vrp["Nb of vehicles"] * vehicle_cost_van

    total_fuel_cost_vrp = scenario_vrp["Distance"] * fuel_consup_van * fuel_cost / 100

    total_cc_cost_van = scenario_vrp["Distance"] * CC_cost_van / 100

    total_ph_cost_van = scenario_vrp["Distance"] * PH_cost_van / 100

    total_time_vrp = scenario_vrp["Average distance per delivery"] * scenario_vrp[
        "Average nb of deliveries per vehicles"] * 60 / speed_van

    total_dropoff_vrp = scenario_vrp["Average nb of deliveries per vehicles"] * dropoff_duration_van

    total_pm_vrp = scenario_vrp["Distance"] * pm_van



    # MDVRP scenario
    total_driver_cost_mdvrp = driver_cost_truck

    total_vehicule_cost_mdvrp = vehicle_cost_truck

    total_fuel_cost_mdvrp = scenario_mdvrp["Vehicle distance"] * fuel_consup_truck * fuel_cost / 100

    total_courriers_cost_mdvrp = nb_deliveries * courrier_cost

    total_cc_cost_truck = scenario_mdvrp["Vehicle distance"] * CC_cost_truck / 100

    total_ph_cost_truck = scenario_mdvrp["Vehicle distance"] * PH_cost_truck / 100

    total_time_mdvrp_truck = (scenario_mdvrp["Vehicle distance"] / 5) * 60 / speed_truck

    total_time_mdvrp_bike = scenario_mdvrp["Vehicle distance"] * 60 / speed_bike

    total_dropoff_mdvrp_truck = dropoff_duration_truck

    total_dropoff_mdvrp_bike = nb_deliveries /scenario_mdvrp["Nb couriers"] * dropoff_duration_bike

    total_pm_mdvrp = scenario_mdvrp["Vehicle distance"] * pm_truck

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # KPI : COST


    ax1.bar(0, total_driver_cost_vrp, color="#ef8a62")
    ax1.bar(0, total_vehicule_cost_vrp, bottom=total_driver_cost_vrp, color="#e06377")
    ax1.bar(0, total_fuel_cost_vrp,
            bottom=total_driver_cost_vrp + total_vehicule_cost_vrp,
            color="#0571b0")
    ax1.bar(0, 0,
            bottom=total_driver_cost_vrp + total_vehicule_cost_vrp + total_fuel_cost_vrp,
            color="#a491ba")
    ax1.bar(0, total_cc_cost_van,
            bottom=total_driver_cost_vrp + total_vehicule_cost_vrp + total_fuel_cost_vrp,
            color="green")
    ax1.bar(0, total_ph_cost_van,
            bottom=total_driver_cost_vrp + total_vehicule_cost_vrp + total_fuel_cost_vrp + total_cc_cost_van,
            color="black")

    ax1.bar(1, total_driver_cost_mdvrp, color="#ef8a62")
    ax1.bar(1, total_vehicule_cost_mdvrp,
            bottom=total_driver_cost_mdvrp,
            color="#e06377")
    ax1.bar(1, total_fuel_cost_mdvrp,
            bottom=total_driver_cost_mdvrp + total_vehicule_cost_mdvrp,
            color="#0571b0")
    ax1.bar(1, total_courriers_cost_mdvrp,
            bottom=total_driver_cost_mdvrp + total_vehicule_cost_mdvrp + total_fuel_cost_mdvrp,
            color="#a491ba")
    ax1.bar(1, total_cc_cost_truck,
            bottom=total_driver_cost_mdvrp + total_vehicule_cost_mdvrp + total_courriers_cost_mdvrp + total_fuel_cost_mdvrp,
            color="green")
    ax1.bar(1, total_ph_cost_truck,
            bottom=total_driver_cost_mdvrp + total_vehicule_cost_mdvrp + total_courriers_cost_mdvrp + total_fuel_cost_mdvrp + total_cc_cost_truck,
            color="black")

    ax1.set_xticklabels(["", "VRP", "", "2EVRP"])
    ax1.set_title("Total cost\n (in €)\n")
    ax1.legend(('Drivers', 'Vehicles', 'Fuel', "Bike", 'Climate Change', "Public Health"),
               bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    # KPI : TIME

    ax2.set_title("Average tour time\n (in min)\n")
    ax2.set_xticklabels(["", "VRP", "", "2EVRP"])
    ax2.bar(0, total_time_vrp, color="#e06377")
    ax2.bar(0, total_dropoff_vrp, bottom=total_time_vrp, color="#f7d4da")
    ax2.bar(0, 0, bottom=total_time_vrp + total_dropoff_vrp, color="#a491ba")
    ax2.bar(0, 0, bottom=total_time_vrp + total_dropoff_vrp, color="#cbc0d8")

    ax2.bar(1, total_time_mdvrp_truck, color="#e06377")
    ax2.bar(1, total_dropoff_mdvrp_truck, bottom=total_time_mdvrp_truck, color="#f7d4da")
    ax2.bar(1, total_time_mdvrp_bike, bottom=total_time_mdvrp_truck + total_dropoff_mdvrp_truck, color="#a491ba")
    ax2.bar(1, total_dropoff_mdvrp_bike, bottom=total_time_mdvrp_truck + total_time_mdvrp_bike + total_dropoff_mdvrp_truck, color="#cbc0d8")

    ax2.legend(("Truck trip time", "Truck dropoff time", "Bike trip time",  "Bike dropoff time"), bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    # KPI : PM EMISSION

    ax3.set_title("Total PM emission\n (in g)\n")
    ax3.set_xticklabels(["", "VRP", "", "2EVRP"])
    ax3.bar(0, total_pm_vrp, color="#e06377")
    ax3.bar(1, total_pm_mdvrp, color="#a491ba")

    fig.subplots_adjust(wspace=0.95)

    # Table

    results = PrettyTable()

    results.field_names = ["Output", "VRP", "2EVRP"]
    results.add_row(["Number of drivers", scenario_vrp["Nb of vehicles"].values[0], 1])
    results.add_row(["Number of courriers", 0, scenario_mdvrp["Nb couriers"].values[0]])
    results.add_row(["Total distance with vehicles (km)", scenario_vrp["Distance"].values[0],
                     scenario_mdvrp["Vehicle distance"].values[0]])


    print(results)
