import pandas as pd
import numpy as np
from pulp import (
    LpProblem, LpMinimize, LpVariable, lpSum, LpBinary,
    LpStatus, value, PULP_CBC_CMD
)


def solve_vrp(nodes_df, requests_df, vehicles_df, time_matrix):
    """
    Główny solver VRP z oknami czasowymi i limitem czasu transportu.
    Zwraca:
      - objective
      - status
      - selected_arcs: łuki x[k,i,j] = 1
      - routes: trasy w postaci (vehicle_id, stop_order, node)
      - lab_assignments: przypisania labów z czasami i slack
      - lifetime: tabela lifetime / slack dla każdej próbki
    """

    # ============================================================
    # 1. PRZYGOTOWANIE DANYCH
    # ============================================================

    time_matrix = time_matrix.copy()
    time_matrix.index = time_matrix.index.astype(int)
    time_matrix.columns = time_matrix.columns.astype(int)

    requests_small = requests_df.copy()
    vehicles_small = vehicles_df.copy()

    hospital_id = 0
    start_node = -1
    end_node = -2

    lab_nodes = requests_small["lab_node_id"].astype(int).tolist()
    vehicles = vehicles_small["vehicle_id"].astype(int).tolist()

    all_nodes = [start_node] + lab_nodes + [end_node]

    ready_time = dict(zip(requests_small["lab_node_id"], requests_small["ready_time"]))
    due_time = dict(zip(requests_small["lab_node_id"], requests_small["due_time"]))
    service_time = dict(zip(requests_small["lab_node_id"], requests_small["service_time"]))
    max_transport_time = dict(zip(requests_small["lab_node_id"], requests_small["max_transport_time"]))

    shift_start = dict(zip(vehicles_small["vehicle_id"], vehicles_small["shift_start"]))
    shift_end = dict(zip(vehicles_small["vehicle_id"], vehicles_small["shift_end"]))

    service_time[start_node] = 0
    service_time[end_node] = 0

    M = 10000
    EPS_VEHICLE = 0.001

    # ============================================================
    # 2. MACIERZ KOSZTÓW (CZASY PRZEJAZDU) Z UWZGLĘDNIENIEM START/END
    # ============================================================

    travel_cost = {}

    for i in all_nodes:
        for j in all_nodes:
            if i == j:
                continue
            if j == start_node:
                continue   # do startu nie wjeżdżamy
            if i == end_node:
                continue   # z końca nie wyjeżdżamy

            real_i = hospital_id if i in [start_node, end_node] else i
            real_j = hospital_id if j in [start_node, end_node] else j

            travel_cost[(i, j)] = float(time_matrix.loc[real_i, real_j])

    arcs = [(i, j) for (i, j) in travel_cost.keys()]

    # ============================================================
    # 3. MODEL
    # ============================================================

    model = LpProblem("VRP_Biological_Samples_No_Spoke_StartEndSplit", LpMinimize)

    # ------------------------------------------------------------
    # Zmienne decyzyjne
    # ------------------------------------------------------------

    # x[k,i,j] = 1, jeśli pojazd k jedzie z i do j
    x = LpVariable.dicts(
        "x",
        ((k, i, j) for k in vehicles for (i, j) in arcs),
        cat=LpBinary
    )

    # t[k,i] = czas rozpoczęcia obsługi węzła i przez pojazd k
    t = LpVariable.dicts(
        "t",
        ((k, i) for k in vehicles for i in all_nodes),
        lowBound=0
    )

    # u[k,i] = 1, jeśli pojazd k obsługuje laboratorium i
    u = LpVariable.dicts(
        "u",
        ((k, i) for k in vehicles for i in lab_nodes),
        cat=LpBinary
    )

    # y[k] = 1, jeśli pojazd k jest użyty
    y = LpVariable.dicts(
        "y",
        (k for k in vehicles),
        cat=LpBinary
    )

    # ============================================================
    # 4. FUNKCJA CELU
    # ============================================================

    model += (
        lpSum(
            travel_cost[(i, j)] * x[(k, i, j)]
            for k in vehicles
            for (i, j) in arcs
        )
        + EPS_VEHICLE * lpSum(y[k] for k in vehicles)
    ), "Minimize_Travel_Time_And_Vehicle_Use"

    # ============================================================
    # 5. KAŻDE LAB ODWIEDZONE DOKŁADNIE RAZ
    # ============================================================

    for i in lab_nodes:
        # dokładnie jedno wyjście z labu łącznie po wszystkich pojazdach
        model += lpSum(
            x[(k, i, j)]
            for k in vehicles
            for j in all_nodes
            if i != j and (i, j) in travel_cost
        ) == 1

        # dokładnie jedno wejście do labu łącznie po wszystkich pojazdach
        model += lpSum(
            x[(k, j, i)]
            for k in vehicles
            for j in all_nodes
            if i != j and (j, i) in travel_cost
        ) == 1

    # ============================================================
    # 6. START / KONIEC TRASY I UŻYCIE POJAZDU
    # ============================================================

    for k in vehicles:
        # jeśli pojazd użyty, to dokładnie raz wyjeżdża ze startu
        model += lpSum(
            x[(k, start_node, j)]
            for j in lab_nodes
            if (start_node, j) in travel_cost
        ) == y[k]

        # jeśli pojazd użyty, to dokładnie raz wjeżdża do end_node
        model += lpSum(
            x[(k, i, end_node)]
            for i in lab_nodes
            if (i, end_node) in travel_cost
        ) == y[k]

    # ============================================================
    # 7. FLOW CONSERVATION W LABORATORIACH
    # ============================================================

    for k in vehicles:
        for i in lab_nodes:
            model += (
                lpSum(
                    x[(k, j, i)]
                    for j in all_nodes
                    if j != i and (j, i) in travel_cost
                )
                ==
                lpSum(
                    x[(k, i, j)]
                    for j in all_nodes
                    if j != i and (i, j) in travel_cost
                )
            )

    # ============================================================
    # 8. ZAKAZY DLA START/END
    # ============================================================

    for k in vehicles:
        # do start_node nic nie wjeżdża
        model += lpSum(
            x[(k, i, start_node)]
            for i in all_nodes
            if i != start_node and (i, start_node) in travel_cost
        ) == 0

        # z end_node nic nie wyjeżdża
        model += lpSum(
            x[(k, end_node, j)]
            for j in all_nodes
            if j != end_node and (end_node, j) in travel_cost
        ) == 0

    # ============================================================
    # 9. POWIĄZANIE u Z ŁUKAMI
    # ============================================================

    for k in vehicles:
        for i in lab_nodes:
            model += u[(k, i)] == lpSum(
                x[(k, i, j)]
                for j in all_nodes
                if j != i and (i, j) in travel_cost
            )

    # ============================================================
    # 10. OKNA CZASOWE
    # ============================================================

    for k in vehicles:
        for i in lab_nodes:
            model += t[(k, i)] >= ready_time[i] - M * (1 - u[(k, i)])
            model += t[(k, i)] <= due_time[i] + M * (1 - u[(k, i)])

    # ============================================================
    # 11. CZAS STARTU I KOŃCA ZMIANY
    # ============================================================

    for k in vehicles:
        model += t[(k, start_node)] == shift_start[k]
        model += t[(k, end_node)] <= shift_end[k]

    # ============================================================
    # 12. PROPAGACJA CZASU
    # ============================================================

    for k in vehicles:
        for i in all_nodes:
            for j in all_nodes:
                if i == j:
                    continue
                if (i, j) not in travel_cost:
                    continue
                if j == start_node:
                    continue
                if i == end_node:
                    continue

                serv_i = service_time.get(i, 0)

                model += (
                    t[(k, j)] >= t[(k, i)] + serv_i + travel_cost[(i, j)]
                    - M * (1 - x[(k, i, j)])
                )

    # ============================================================
    # 13. OGRANICZENIE MAKS. CZASU TRANSPORTU PRÓBKI
    # ============================================================

    for k in vehicles:
        for i in lab_nodes:
            model += (
                t[(k, end_node)] - (t[(k, i)] + service_time[i])
                <= max_transport_time[i] + M * (1 - u[(k, i)])
            )

    # ============================================================
    # 14. ROZWIĄZANIE
    # ============================================================

    solver = PULP_CBC_CMD(msg=True, timeLimit=300)
    model.solve(solver)

    # ============================================================
    # 15. WYBRANE ŁUKI x[k,i,j] = 1
    # ============================================================

    selected_arcs = []
    for k in vehicles:
        for (i, j) in arcs:
            val = value(x[(k, i, j)])
            if val is not None and val > 0.5:
                selected_arcs.append({
                    "vehicle_id": k,
                    "from_node": i,
                    "to_node": j,
                    "travel_time_min": travel_cost[(i, j)]
                })

    selected_arcs_df = pd.DataFrame(selected_arcs)

    # ============================================================
    # 16. ODTWORZENIE TRAS Z ŁUKÓW
    # ============================================================

    def build_vehicle_route(arcs_df, vehicle_id, start_node=-1, end_node=-2):
        sub = arcs_df[arcs_df["vehicle_id"] == vehicle_id].copy()
        if sub.empty:
            return []

        arc_map = dict(zip(sub["from_node"], sub["to_node"]))

        route = [start_node]
        current = start_node
        visited = set([start_node])

        while current in arc_map:
            nxt = arc_map[current]
            route.append(nxt)

            if nxt == end_node:
                break

            if nxt in visited:
                # cykl – przerywamy
                break

            visited.add(nxt)
            current = nxt

        return route

    routes_rows = []
    for k in vehicles:
        route = build_vehicle_route(selected_arcs_df, k)
        for order, node in enumerate(route):
            routes_rows.append({
                "vehicle_id": k,
                "stop_order": order,
                "node": node
            })

    routes_df = pd.DataFrame(routes_rows)

    # ============================================================
    # 17. PRZYPISANIA LABÓW + SLACK
    # ============================================================

    lab_assignment_rows = []
    for k in vehicles:
        for i in lab_nodes:
            if value(u[(k, i)]) > 0.5:

                pickup_time = value(t[(k, i)])
                delivery_time = value(t[(k, end_node)])
                transport_time = delivery_time - (pickup_time + service_time[i])
                slack = max_transport_time[i] - transport_time

                lab_assignment_rows.append({
                    "vehicle_id": k,
                    "lab_node_id": i,
                    "visit_time": pickup_time,
                    "ready_time": ready_time[i],
                    "due_time": due_time[i],
                    "service_time": service_time[i],
                    "max_transport_time": max_transport_time[i],
                    "delivery_time": delivery_time,
                    "transport_time": transport_time,
                    "slack": slack
                })

    lab_assignment_df = pd.DataFrame(lab_assignment_rows)

    # ============================================================
    # 18. LIFETIME / SLA – TABELA POMOCNICZA
    # ============================================================

    lifetime_rows = []

    for k in vehicles:
        for i in lab_nodes:
            if value(u[(k, i)]) > 0.5:
                pickup_time = value(t[(k, i)])
                delivery_time = value(t[(k, end_node)])

                transport_time = delivery_time - (pickup_time + service_time[i])
                max_time = max_transport_time[i]
                slack = max_time - transport_time

                lifetime_rows.append({
                    "vehicle": k,
                    "lab": i,
                    "pickup_time": pickup_time,
                    "delivery_time": delivery_time,
                    "transport_time": transport_time,
                    "max_allowed": max_time,
                    "slack": slack
                })

    lifetime_df = pd.DataFrame(lifetime_rows)

    # ============================================================
    # 18A. UŻYCIE POJAZDÓW (y[k])
    # ============================================================

    vehicle_usage_rows = []
    for k in vehicles:
        y_val = value(y[k]) or 0
        vehicle_usage_rows.append({
            "vehicle_id": k,
            "used": int(round(y_val))
        })

    vehicle_usage_df = pd.DataFrame(vehicle_usage_rows)

    # ============================================================
    # 18B. CZASY ODWIEDZIN WSZYSTKICH WĘZŁÓW
    # ============================================================

    node_times_rows = []
    for k in vehicles:
        for i in all_nodes:
            node_times_rows.append({
                "vehicle_id": k,
                "node": i,
                "time": value(t[(k, i)])
            })

    node_times_df = pd.DataFrame(node_times_rows)

    # ============================================================
    # 18C. MANUAL SUM OF TRAVEL TIMES
    # ============================================================

    manual_travel_sum = selected_arcs_df["travel_time_min"].sum()

    # ============================================================
    # 19. ZWROT WYNIKÓW
    # ============================================================

    

    return {
        "objective": value(model.objective),
        "status": LpStatus[model.status],
        "selected_arcs": selected_arcs_df,
        "routes": routes_df,
        "lab_assignments": lab_assignment_df,
        "lifetime": lifetime_df,
        "vehicle_usage": vehicle_usage_df,      # NOWE
        "node_times": node_times_df,            # NOWE
        "manual_travel_sum": manual_travel_sum  # OPCJONALNE
    }
