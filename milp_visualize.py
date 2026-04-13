import folium
import networkx as nx
import osmnx as ox
import pandas as pd

def visualize_routes(
    nodes_df,
    routes,
    G,
    OUT_DIR,
    map_filename="milp_routes_map.html"
):
    """
    Tworzy mapę tras MILP i zapisuje ją do OUT_DIR/map_filename.
    Zwraca DataFrame z podsumowaniem tras.
    """

    # ============================================================
    # 1. PRZYGOTOWANIE MAPOWANIA NODE_ID -> WSPÓŁRZĘDNE / OSM NODE
    # ============================================================

    if "osmnx_node" not in nodes_df.columns:
        nearest_nodes = ox.distance.nearest_nodes(
            G,
            X=nodes_df["lon"].values,
            Y=nodes_df["lat"].values
        )
        nodes_df["osmnx_node"] = nearest_nodes

    node_lookup = nodes_df.set_index("node_id").to_dict("index")

    hospital_id = 0
    start_node = -1
    end_node = -2

    def real_node_id(node):
        if node in [start_node, end_node]:
            return hospital_id
        return node

    # ============================================================
    # 2. MAPA BAZOWA
    # ============================================================

    center_lat = nodes_df["lat"].mean()
    center_lon = nodes_df["lon"].mean()

    m_milp = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles="OpenStreetMap"
    )

    # ============================================================
    # 3. RYSOWANIE PUNKTÓW
    # ============================================================

    for _, row in nodes_df.iterrows():
        if row["type"] == "hospital":
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=9,
                popup=f"<b>{row['name']}</b>",
                tooltip="Hospital",
                color="red",
                fill=True,
                fill_opacity=1.0
            ).add_to(m_milp)
        else:
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=6,
                popup=f"{row['name']} (node_id={row['node_id']})",
                tooltip=f"{row['name']}",
                color="blue",
                fill=True,
                fill_opacity=0.9
            ).add_to(m_milp)

            folium.Marker(
                [row["lat"], row["lon"]],
                icon=folium.DivIcon(
                    html=f"""
                    <div style="font-size: 10pt; color: black; font-weight: bold;">
                        {int(row['node_id'])}
                    </div>
                    """
                )
            ).add_to(m_milp)

    # ============================================================
    # 4. KOLORY DLA POJAZDÓW
    # ============================================================

    vehicle_colors = {
        1: "green",
        2: "purple",
        3: "orange",
        4: "darkred",
        5: "cadetblue",
        6: "yellow"
    }

    # ============================================================
    # 5. RYSOWANIE TRAS POJAZDÓW
    # ============================================================

    route_summary = []

    for vehicle_id, route in routes.items():
        if not route or len(route) < 2:
            continue

        color = vehicle_colors.get(vehicle_id, "blue")

        total_length_m = 0
        total_time_min = 0

        for idx in range(len(route) - 1):
            from_model_node = route[idx]
            to_model_node = route[idx + 1]

            from_real = real_node_id(from_model_node)
            to_real = real_node_id(to_model_node)

            from_osm = node_lookup[from_real]["osmnx_node"]
            to_osm = node_lookup[to_real]["osmnx_node"]

            path = nx.shortest_path(G, from_osm, to_osm, weight="travel_time")

            seg_length_m = nx.shortest_path_length(G, from_osm, to_osm, weight="length")
            seg_time_sec = nx.shortest_path_length(G, from_osm, to_osm, weight="travel_time")

            total_length_m += seg_length_m
            total_time_min += seg_time_sec / 60.0

            path_coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in path]

            folium.PolyLine(
                locations=path_coords,
                weight=5,
                opacity=0.85,
                color=color,
                tooltip=f"Vehicle {vehicle_id}: {from_model_node} → {to_model_node}"
            ).add_to(m_milp)

        # oznaczenia punktów
        for stop_order, model_node in enumerate(route):
            real_id = real_node_id(model_node)
            row = node_lookup[real_id]

            if model_node == start_node:
                label = f"V{vehicle_id}-START"
            elif model_node == end_node:
                label = f"V{vehicle_id}-END"
            else:
                label = f"V{vehicle_id}-{stop_order}"

            folium.Marker(
                [row["lat"], row["lon"]],
                icon=folium.DivIcon(
                    html=f"""
                    <div style="
                        font-size: 9pt;
                        color: {color};
                        font-weight: bold;
                        background-color: white;
                        border: 1px solid {color};
                        border-radius: 4px;
                        padding: 1px 3px;
                    ">
                        {label}
                    </div>
                    """
                )
            ).add_to(m_milp)

        route_summary.append({
            "vehicle_id": vehicle_id,
            "route": " -> ".join(map(str, route)),
            "distance_km": round(total_length_m / 1000.0, 2),
            "travel_time_min": round(total_time_min, 2)
        })

    # ============================================================
    # 6. PODSUMOWANIE
    # ============================================================

    route_summary_df = pd.DataFrame(route_summary)

    # ============================================================
    # 7. ZAPIS MAPY
    # ============================================================

    map_path = OUT_DIR / map_filename
    m_milp.save(map_path)

    print("\nZapisano mapę:", map_path.resolve())

    return route_summary_df
