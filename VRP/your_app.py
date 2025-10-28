import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import pydeck as pdk
import plotly.express as px

# Cache the data to prevent reloading on every interaction
@st.cache_data
def load_data():
    routes_df = pd.read_csv('routes_distance.csv')
    vehicles_df = pd.read_csv('vehicle_fleet.csv')
    return routes_df, vehicles_df

def preprocess_data(routes_df, vehicles_df):
    """Cleans and preprocesses the data."""
    # Clean routes data
    routes_df['Weather_Impact'] = routes_df['Weather_Impact'].fillna('No_Impact')
    routes_df.replace('None', 'No_Impact', inplace=True)

    # Feature Engineering: Calculate costs and penalties
    fuel_price_inr = 100  # Assume INR 100 per liter
    
    # Basic costs
    routes_df['Fuel_Cost_INR'] = routes_df['Fuel_Consumption_L'] * fuel_price_inr
    routes_df['Total_Cost_INR'] = routes_df['Fuel_Cost_INR'] + routes_df['Toll_Charges_INR']
    
    # Add weather impact penalties
    weather_penalties = {
        'No_Impact': 1.0,
        'Light_Rain': 1.2,  # 20% penalty
        'Heavy_Rain': 1.5,  # 50% penalty
        'Fog': 1.4         # 40% penalty
    }
    routes_df['Weather_Penalty'] = routes_df['Weather_Impact'].map(weather_penalties)
    
    # Add time delay costs (assume INR 500 per hour of delay)
    routes_df['Delay_Cost_INR'] = (routes_df['Traffic_Delay_Minutes'] / 60) * 500
    
    # Calculate efficiency score (lower is better)
    routes_df['Efficiency_Score'] = (
        routes_df['Total_Cost_INR'] * routes_df['Weather_Penalty'] + 
        routes_df['Delay_Cost_INR']
    ) / routes_df['Distance_KM']

    # Process vehicle data
    # Add vehicle type efficiency ratings
    type_specs = {
        'Express_Bike': {'weight_limit': 50, 'weather_resistance': 0.5},
        'Small_Van': {'weight_limit': 1000, 'weather_resistance': 0.8},
        'Medium_Truck': {'weight_limit': 4000, 'weather_resistance': 0.9},
        'Large_Truck': {'weight_limit': 10000, 'weather_resistance': 1.0},
        'Refrigerated': {'weight_limit': 3000, 'weather_resistance': 0.85}
    }
    
    vehicles_df['Weight_Limit'] = vehicles_df['Vehicle_Type'].map(lambda x: type_specs[x]['weight_limit'])
    vehicles_df['Weather_Resistance'] = vehicles_df['Vehicle_Type'].map(lambda x: type_specs[x]['weather_resistance'])
    
    return routes_df, vehicles_df

def main():
    st.set_page_config(layout="wide")
    st.title('Route Optimization')
    routes_df, vehicles_df = load_data()
    routes_df, vehicles_df = preprocess_data(routes_df, vehicles_df)
    
    # vehicle type characteristics
    vehicle_type_weather_resistance = {
        'Express_Bike': 0.5,    # Most affected by weather
        'Small_Van': 0.8,       # Good weather resistance
        'Medium_Truck': 0.9,    # Very good weather resistance
        'Large_Truck': 1.0,     # Best weather resistance
        'Refrigerated': 0.85    # Good weather resistance but sensitive to extreme conditions
    }

    # coordinates for map 
    locations_coords = {
        'Kolkata': {'lat': 22.5726, 'lon': 88.3639},
        'Hyderabad': {'lat': 17.3850, 'lon': 78.4867},
        'Mumbai': {'lat': 19.0760, 'lon': 72.8777},
        'Pune': {'lat': 18.5204, 'lon': 73.8567},
        'Ahmedabad': {'lat': 23.0225, 'lon': 72.5714},
        'Chennai': {'lat': 13.0827, 'lon': 80.2707},
        'Dubai': {'lat': 25.2048, 'lon': 55.2708},
        'Bangalore': {'lat': 12.9716, 'lon': 77.5946},
        'Delhi': {'lat': 28.7041, 'lon': 77.1025},
        'Hong Kong': {'lat': 22.3193, 'lon': 114.1694},
        'Bangkok': {'lat': 13.7563, 'lon': 100.5018},
        'Singapore': {'lat': 1.3521, 'lon': 103.8198}
    }
    
    routes_df['Start'] = routes_df['Route'].apply(lambda x: x.split('-')[0])
    routes_df['End'] = routes_df['Route'].apply(lambda x: x.split('-')[1])


    st.sidebar.header('User Input')
    
    # Get unique origins and destinations from the Route column
    route_locations = set()
    for route in routes_df['Route']:
        locations = route.split('-')
        route_locations.update(locations)
    
    origin = st.sidebar.selectbox('Origin', sorted(list(route_locations)))
    destination = st.sidebar.selectbox('Destination', sorted(list(route_locations)))
    vehicle_type = st.sidebar.selectbox('Vehicle Type', vehicles_df['Vehicle_Type'].unique())

    if st.sidebar.button('Find Best Route'):
        # Create a graph for routing
        G = nx.from_pandas_edgelist(routes_df, 'Start', 'End', ['Total_Cost_INR', 'Distance_KM'])

        try:
            def estimate_route_metrics(start, end):
                """Estimate route metrics for missing routes based on coordinates."""
                from math import radians, sin, cos, sqrt, atan2
                
                # Calculate great circle distance
                lat1, lon1 = radians(locations_coords[start]['lat']), radians(locations_coords[start]['lon'])
                lat2, lon2 = radians(locations_coords[end]['lat']), radians(locations_coords[end]['lon'])
                
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                c = 2 * atan2(sqrt(a), sqrt(1-a))
                R = 6371  # Earth's radius in km
                distance = R * c

                # Calculate averages from existing routes
                existing_routes = routes_df[routes_df['Distance_KM'].notna()]
                avg_fuel_per_km = existing_routes['Fuel_Consumption_L'].sum() / existing_routes['Distance_KM'].sum()
                avg_toll_per_km = existing_routes[existing_routes['Toll_Charges_INR'] > 0]['Toll_Charges_INR'].sum() / \
                                 existing_routes[existing_routes['Toll_Charges_INR'] > 0]['Distance_KM'].sum()
                
                # For international routes, no toll charges
                is_international = any(city in [start, end] for city in ['Dubai', 'Bangkok', 'Singapore', 'Hong Kong'])
                estimated_toll = 0 if is_international else distance * avg_toll_per_km
                estimated_fuel = distance * avg_fuel_per_km
                
                # Calculate estimated costs
                fuel_cost = estimated_fuel * 100  # fuel_price_inr = 100
                delay_cost = (30 / 60) * 500  # Assume 30 minutes average delay
                total_cost = fuel_cost + estimated_toll + delay_cost
                
                # Create efficiency score similar to actual routes
                efficiency_score = total_cost / distance
                
                return efficiency_score

            # Create an edge weight function that considers multiple factors
            def edge_weight(u, v, d):
                route_data = routes_df[(routes_df['Start'] == u) & (routes_df['End'] == v)]

                if route_data.empty:
                    # For missing routes, estimate the metrics
                    base_cost = estimate_route_metrics(u, v)
                    # Add a small penalty for estimated routes to prefer known routes
                    base_cost *= 1.1  # 10% penalty for estimated routes
                else:
                    route_data = route_data.iloc[0]
                    base_cost = route_data['Efficiency_Score']
                    
                    # Add weather-specific penalties based on vehicle type
                    if route_data['Weather_Impact'] != 'No_Impact':
                        weather_factor = 1 + (1 - vehicle_type_weather_resistance.get(vehicle_type, 0.8))
                        base_cost *= weather_factor

                return base_cost


            def heuristic(u, v):
                """Heuristic for A* algorithm: great-circle distance."""
                from math import radians, sin, cos, sqrt, atan2
                
                lat1, lon1 = radians(locations_coords[u]['lat']), radians(locations_coords[u]['lon'])
                lat2, lon2 = radians(locations_coords[v]['lat']), radians(locations_coords[v]['lon'])
                
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                c = 2 * atan2(sqrt(a), sqrt(1-a))
                R = 6371  # Earth's radius
                return R * c

            # Find all simple paths between origin and destination
            all_paths = list(nx.all_simple_paths(G, source=origin, target=destination, cutoff=3))
            
            # Evaluate each path to find the most efficient one
            path_scores = []
            for p in all_paths:
                path_cost = 0
                for i in range(len(p)-1):
                    path_cost += edge_weight(p[i], p[i+1], None)
                path_scores.append((p, path_cost))
            
            # Select the path with minimum cost
            path = min(path_scores, key=lambda x: x[1])[0]
            path_edges = list(zip(path, path[1:]))

            # Get route details for the path, including estimated routes
            path_details = []
            for start, end in path_edges:
                route_data = routes_df[(routes_df['Start'] == start) & (routes_df['End'] == end)]
                
                if route_data.empty:
                    # Calculate metrics for missing route
                    from math import radians, sin, cos, sqrt, atan2
                    
                    # Calculate distance using Haversine formula
                    lat1, lon1 = radians(locations_coords[start]['lat']), radians(locations_coords[start]['lon'])
                    lat2, lon2 = radians(locations_coords[end]['lat']), radians(locations_coords[end]['lon'])
                    
                    dlat = lat2 - lat1
                    dlon = lon2 - lon1
                    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                    c = 2 * atan2(sqrt(a), sqrt(1-a))
                    R = 6371  # Earth's radius in km
                    distance = R * c
                    
                    # Calculate averages from existing routes for estimation
                    avg_fuel_per_km = routes_df['Fuel_Consumption_L'].sum() / routes_df['Distance_KM'].sum()
                    avg_toll_per_km = routes_df[routes_df['Toll_Charges_INR'] > 0]['Toll_Charges_INR'].sum() / \
                                    routes_df[routes_df['Toll_Charges_INR'] > 0]['Distance_KM'].sum()
                    
                    # Create estimated route details
                    is_international = any(city in [start, end] for city in ['Dubai', 'Bangkok', 'Singapore', 'Hong Kong'])
                    estimated_details = pd.Series({
                        'Route': f"{start}-{end}",
                        'Distance_KM': distance,
                        'Fuel_Consumption_L': distance * avg_fuel_per_km,
                        'Toll_Charges_INR': 0 if is_international else distance * avg_toll_per_km,
                        'Traffic_Delay_Minutes': 30,  # Average delay
                        'Weather_Impact': 'No_Impact',
                        'Start': start,
                        'End': end,
                        'Estimated': True  # Flag to indicate this is an estimated route
                    })
                    
                    # Calculate costs for estimated route
                    estimated_details['Fuel_Cost_INR'] = estimated_details['Fuel_Consumption_L'] * 100  # fuel_price_inr = 100
                    estimated_details['Total_Cost_INR'] = estimated_details['Fuel_Cost_INR'] + estimated_details['Toll_Charges_INR']
                    estimated_details['Delay_Cost_INR'] = (estimated_details['Traffic_Delay_Minutes'] / 60) * 500
                    
                    path_details.append(estimated_details)
                else:
                    details = route_data.iloc[0]
                    details['Estimated'] = False  # Flag to indicate this is a real route
                    path_details.append(details)
            
            # Calculate totals
            total_cost = sum(d['Total_Cost_INR'] for d in path_details)
            total_distance = sum(d['Distance_KM'] for d in path_details)

            # Find available vehicles and assign the most suitable one
            available_vehicles = vehicles_df[(vehicles_df['Vehicle_Type'] == vehicle_type) & (vehicles_df['Status'] == 'Available')]
            
            if not available_vehicles.empty:
                # Calculate total route distance
                total_distance = sum(d['Distance_KM'] for d in path_details)
                
                # Score each vehicle based on multiple criteria
                def score_vehicle(vehicle):
                    score = 0
                    # Prefer vehicles with higher fuel efficiency for longer routes
                    score += vehicle['Fuel_Efficiency_KM_per_L'] * (total_distance/1000)  
                    # Prefer newer vehicles (less age)
                    score += (10 - vehicle['Age_Years']) * 0.5  # Assume 10 years as max age
                    # Prefer vehicles with better emissions
                    score -= vehicle['CO2_Emissions_Kg_per_KM'] * 100
                    return score
                
                # Add scores to available vehicles
                available_vehicles['Score'] = available_vehicles.apply(score_vehicle, axis=1)
                # Get the best vehicle based on scoring
                assigned_vehicle = available_vehicles.nlargest(1, 'Score').iloc[0]
                
                # Route Overview
                st.subheader(f"Optimal Route: {origin} to {destination}")
                
                # Create a more detailed route display
                route_steps = []
                for i, (start, end) in enumerate(path_edges):
                    route_data = routes_df[(routes_df['Start'] == start) & (routes_df['End'] == end)]
                    is_direct = not route_data.empty
                    step_text = f"{i+1}. {start} → {end}"
                    if not is_direct:
                        step_text += " (Estimated Route)"
                    route_steps.append(step_text)
                
                st.markdown("### Route Steps")
                for step in route_steps:
                    st.write(step)

                # Route Summary Section
                st.subheader("Route Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(label="Total Distance (KM)", value=f"{total_distance:.2f}")
                with col2:
                    st.metric(label="Total Cost (INR)", value=f"₹{total_cost:.2f}")
                with col3:
                    avg_cost_per_km = total_cost / total_distance
                    st.metric(label="Cost per KM", value=f"₹{avg_cost_per_km:.2f}")

                # Route Details Section
                st.subheader("📊 Route Analysis")
                
                # Create a DataFrame for route segments for use in all tabs
                route_df = pd.DataFrame(path_details)
                route_df['Segment'] = [f"{start}-{end}" for start, end in zip(route_df['Start'], route_df['End'])]

                # Create tabs for different visualizations
                tab1, tab2, tab3 = st.tabs(["Cost Breakdown", "Route Details", "Weather Impact"])
                
                with tab1:
                    # Create cost breakdown
                    cost_data = pd.DataFrame({
                        'Cost Type': ['Fuel Cost', 'Toll Charges', 'Delay Cost'],
                        'Amount': [
                            route_df['Fuel_Cost_INR'].sum(), 
                            route_df['Toll_Charges_INR'].sum(),
                            route_df['Delay_Cost_INR'].sum()
                        ]
                    })
                    
                    # Plot cost breakdown
                    fig_cost = px.pie(cost_data, values='Amount', names='Cost Type',
                                    title='Cost Distribution', hole=0.3)
                    st.plotly_chart(fig_cost, use_container_width=True)

                with tab2:
                    # Display detailed route segments
                    st.subheader("Route Segment Details")
                    display_df = route_df[['Segment', 'Distance_KM', 'Fuel_Consumption_L', 'Toll_Charges_INR', 'Traffic_Delay_Minutes', 'Weather_Impact']]
                    st.dataframe(display_df.style.highlight_max(axis=0, subset=['Distance_KM', 'Fuel_Consumption_L', 'Toll_Charges_INR', 'Traffic_Delay_Minutes'], color='lightcoral'))

                with tab3:
                    # Weather impact analysis
                    st.subheader("Weather Conditions Along Route")
                    weather_counts = route_df['Weather_Impact'].value_counts().reset_index()
                    weather_counts.columns = ['Weather Condition', 'Number of Segments']
                    
                    fig_weather = px.pie(weather_counts, 
                                       values='Number of Segments', 
                                       names='Weather Condition',
                                       title='Proportion of Route by Weather Condition',
                                       color_discrete_map={
                                           'No_Impact': 'green',
                                           'Light_Rain': 'yellow',
                                           'Heavy_Rain': 'orange',
                                           'Fog': 'grey'
                                       })
                    st.plotly_chart(fig_weather, use_container_width=True)

                # Vehicle Assignment Section
                st.subheader("Vehicle Assignment")
                vehicle_col1, vehicle_col2 = st.columns(2)
                
                with vehicle_col1:
                    # Vehicle details table
                    st.markdown("### Assigned Vehicle Specifications")
                    vehicle_details = pd.DataFrame({
                        'Specification': ['Vehicle ID', 'Type', 'Fuel Efficiency', 'Capacity', 'Age', 'CO2 Emissions'],
                        'Value': [
                            assigned_vehicle['Vehicle_ID'],
                            assigned_vehicle['Vehicle_Type'],
                            f"{assigned_vehicle['Fuel_Efficiency_KM_per_L']:.2f} KM/L",
                            f"{assigned_vehicle['Capacity_KG']:.2f} KG",
                            f"{assigned_vehicle['Age_Years']:.1f} years",
                            f"{assigned_vehicle['CO2_Emissions_Kg_per_KM']:.3f} Kg/KM"
                        ]
                    })
                    st.table(vehicle_details)
                
                with vehicle_col2:
                    # Show selection criteria scores
                    st.markdown("### Vehicle Selection Analysis")
                    scores_df = available_vehicles[['Vehicle_ID', 'Score']].sort_values('Score', ascending=False)
                    fig_scores = px.bar(scores_df, x='Vehicle_ID', y='Score',
                                      title='Vehicle Suitability Ranking',
                                      color_discrete_sequence=['#0083B8'])
                    fig_scores.update_layout(
                        xaxis_title="Vehicle ID",
                        yaxis_title="Suitability Score",
                        showlegend=False
                    )
                    st.plotly_chart(fig_scores, use_container_width=True)

                # Map Visualization
                st.subheader("🗺️ Route Map")
                
                # Create route path layer
                path_points = []
                for i in range(len(path)-1):
                    start = path[i]
                    end = path[i+1]
                    path_points.append({
                        'start_lat': locations_coords[start]['lat'],
                        'start_lon': locations_coords[start]['lon'],
                        'end_lat': locations_coords[end]['lat'],
                        'end_lon': locations_coords[end]['lon']
                    })
                
                # Create points layer for locations
                points_data = pd.DataFrame([
                    {
                        'lat': locations_coords[loc]['lat'],
                        'lon': locations_coords[loc]['lon'],
                        'name': loc,
                        'color': [255, 0, 0] if loc == origin else [0, 255, 0] if loc == destination else [0, 0, 255]
                    }
                    for loc in path
                ])

                # Create deck layers
                layers = [
                    # Line layer for routes
                    pdk.Layer(
                        'LineLayer',
                        data=pd.DataFrame(path_points),
                        get_source_position=['start_lon', 'start_lat'],
                        get_target_position=['end_lon', 'end_lat'],
                        get_color=[0, 100, 255],
                        get_width=3,
                        pickable=True
                    ),
                    # Scatter layer for locations
                    pdk.Layer(
                        'ScatterplotLayer',
                        data=points_data,
                        get_position=['lon', 'lat'],
                        get_color='color',
                        get_radius=5000,
                        pickable=True
                    )
                ]

                # Set the viewport location
                view_state = pdk.ViewState(
                    latitude=locations_coords[origin]['lat'],
                    longitude=locations_coords[origin]['lon'],
                    zoom=4,
                    pitch=45
                )

                # Render the deck
                deck = pdk.Deck(
                    layers=layers,
                    initial_view_state=view_state,
                    tooltip={
                        'html': '{name}',
                        'style': {
                            'color': 'white'
                        }
                    }
                )
                
                st.pydeck_chart(deck)

            else:
                st.warning(f"No available vehicles of type '{vehicle_type}' found.")
        
        except nx.NetworkXNoPath:
            st.error(f"No route found from {origin} to {destination}.")

    with st.expander("View Raw Data"):
        st.subheader('Route Data')
        st.write(routes_df)
        st.subheader('Vehicle Fleet Data')
        st.write(vehicles_df)


if __name__ == "__main__":
    main()