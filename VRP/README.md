# Vehicle Route Planning (VRP) System

A streamlined route optimization system built with Python and Streamlit that helps plan efficient routes while considering multiple factors such as weather conditions, traffic delays, and vehicle specifications.

## Features

- **Route Optimization**
  - Finds optimal routes between cities using both direct and indirect paths
  - Considers weather conditions and traffic delays
  - Handles international and domestic routes differently
  - Estimates routes for undefined paths using geographical distance

- **Vehicle Assignment**
  - Intelligent vehicle selection based on multiple criteria
  - Considers fuel efficiency, age, and environmental impact
  - Matches vehicle types to route conditions
  - Real-time availability tracking

- **Cost Analysis**
  - Detailed breakdown of costs (fuel, toll, delays)
  - Per-kilometer cost calculation
  - Weather impact cost adjustments
  - Traffic delay cost estimation

- **Interactive Visualization**
  - Interactive map showing complete route
  - Cost distribution charts
  - Route segment analysis
  - Weather impact visualization
  - Vehicle performance comparison

## Dependencies

- Python 3.x
- Streamlit
- Pandas
- NetworkX
- Plotly
- PyDeck

## Setup

1. Install required packages:
```bash
pip install streamlit pandas networkx plotly pydeck
```

2. Run the application:
```bash
streamlit run your_app.py
```

## Data Files

- `routes_distance.csv`: Contains route information between cities
- `vehicle_fleet.csv`: Contains vehicle fleet information

## Usage

1. Select origin and destination cities from the dropdown menus
2. Choose desired vehicle type
3. Click "Find Best Route" to generate optimized route
4. View detailed analysis including:
   - Route segments and total distance
   - Cost breakdown
   - Weather conditions
   - Vehicle assignment details

## Implementation Details

The system uses:
- NetworkX for path finding and graph operations
- Haversine formula for distance estimation
- A* algorithm with custom heuristics for route optimization
- Custom scoring system for vehicle assignment
- PyDeck for route visualization
- Plotly for interactive charts

## Project Structure

```
VRP/
├── your_app.py           # Main application file
├── routes_distance.csv  # Route data
├── vehicle_fleet.csv    # Vehicle fleet data
├── README.md           # Project documentation
└── approach.txt        # Detailed implementation approach
```