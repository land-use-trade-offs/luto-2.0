# VUE_LUTO - Land Use Trade-Offs (LUTO) 2.0 Dashboard

## Overview

VUE_LUTO is a web-based dashboard application for visualizing and analyzing results from the Land Use Trade-Offs (LUTO) 2.0 model. It provides interactive charts, maps, and data exploration tools for understanding the environmental, economic, and social impacts of different land use scenarios in Australia.

## Purpose

The LUTO model is designed to analyze trade-offs between different land uses in Australia, considering factors such as:
- **Economics**: Revenue, costs, and economic indicators
- **Area Analysis**: Land use distribution and changes
- **Greenhouse Gas (GHG) Emissions**: Carbon footprint and climate impacts
- **Water Usage**: Water consumption and management
- **Biodiversity**: Environmental conservation metrics

This dashboard provides an intuitive interface to explore model outputs and understand the implications of different policy scenarios.

## Technology Stack

- **Frontend Framework**: Vue.js 3.5.18 with Composition API
- **Routing**: Vue Router 4.5.1
- **Styling**: Tailwind CSS 3.4.16
- **UI Components**: Element Plus 2.10.4 for enhanced UI components
- **Charts**: Highcharts 12.3.0 with accessibility features
- **Maps**: Leaflet 1.9.4 for interactive Australian region mapping
- **Architecture**: Single Page Application (SPA) with no build process
- **Dependencies**: All libraries are locally hosted in the `lib/` directory for offline use

## Project Structure

```
VUE_LUTO/
├── components/                         # Reusable Vue components
│   ├── chart_container.js              # Highcharts wrapper component
│   ├── helpers.js                      # Utility functions for script/data loading
│   ├── map_geojson.js                  # Interactive map component
│   ├── ranking_cards.js                # Ranking cards component
│   ├── filterable_dropdown.js          # Searchable dropdown component
│   ├── regions_map.js                  # Region selection map component
│   └── sidebar.js                      # Navigation sidebar
├── views/                              # Page components (routes)
│   ├── Home.js                         # Main dashboard with overview
│   ├── Area.js                         # Area analysis view
│   ├── Economics.js                    # Economics analysis view
│   ├── GHG.js                          # Greenhouse Gas analysis view
│   ├── Water.js                        # Water usage analysis view
│   ├── Production.js                   # Production analysis view
│   ├── Settings.js                     # Application settings view
│   └── NotFound.js                     # 404 error page
├── services/                           # Service modules
│   ├── DataService.js                  # Data handling service
│   └── MapService.js                   # Map data and interactions service
├── routes/                             # Routing configuration
│   └── route.js                        # Vue Router setup
├── data/                               # Data files and model outputs
│   ├── chart_option/                   # Chart configuration templates
│   │   ├── Chart_default_options.js    # Default chart styles
│   │   └── chartMemLogOptions.js       # Memory log chart configuration
│   ├── geo/                            # Geographic data (Australian regions)
│   └── Supporting_info.js              # Consolidated model settings and information
├── lib/                                # Local library dependencies
│   ├── Highcharts-12.3.0/              # Highcharts library and modules
│   ├── vue.global.prod_3.5.18.js       # Vue.js library
│   ├── vue-router.global_4.5.1.js      # Vue Router library
│   └── tailwind_3.4.16.js              # Tailwind CSS library
├── assets/                             # Raw data assets (JSON format)
├── dataTransform/                      # Data transformation scripts
│   ├── 01_JSON2JS_dataTrans.py         # JSON to JS conversion utility
│   └── NRM_SIMPLIFY_FILTER/            # Geographic data processing tools
├── resources/                          # Static assets
│   ├── icons.js                        # SVG icons
│   ├── LUTO.png                        # Logo
│   └── Roboto-Light.ttf                # Custom font
├── index.html                          # Main HTML entry point
└── index.js                            # Application bootstrap

```

## Key Features

### 1. Interactive Dashboard (Home View)
- **Overview Charts**: Displays key metrics for different domains (economics, area, GHG, water, biodiversity)
- **Memory Usage Monitoring**: Real-time visualization of model execution memory consumption
- **Parameter Summary**: Searchable list of model run settings and parameters
- **Regional Selection**: Interactive map for selecting Australian regions

### 2. Area Analysis
- Detailed land use area breakdowns
- Temporal analysis of land use changes
- Multiple dataset visualization options

### 3. Interactive Map
- **Australian Regions**: Based on Natural Resource Management (NRM) regions
- **Hover Effects**: Region highlighting and tooltips
- **Region Selection**: Click to select regions for detailed analysis
- **Responsive Design**: Adapts to different screen sizes

### 4. Chart System
- **Highcharts Integration**: Professional-grade interactive charts
- **Export Capabilities**: PNG, JPEG, PDF, and CSV export options
- **Accessibility**: Screen reader support and keyboard navigation
- **Responsive Design**: Charts adapt to container sizes

## Data Architecture

### Dynamic Data Loading
The application uses a custom script loading system (`helpers.js`) that:
- Loads data files on-demand to optimize performance
- Manages script dependencies and loading order
- Provides error handling for failed data loads
- Supports timeout mechanisms for reliable loading

### Data Types
1. **Supporting Info** (`Supporting_info.js`): Consolidated information including model run settings
2. **Chart Data**: Time-series and categorical data organized by region and category
3. **Geographic Data** (`NRM_AUS.js`): GeoJSON data for Australian regions
4. **Chart Options**: Multiple files with specific chart configurations:
   - `Chart_default_options.js`: Default styling and configuration for charts
   - `chartMemLogOptions.js`: Memory log chart specific options

## Component Architecture

### Chart Container (`chart_container.js`)
- Wraps Highcharts functionality in a Vue component
- Manages chart lifecycle (creation, updates, destruction)
- Handles loading states and error conditions
- Supports reactive data updates

### Map Component (`map_geojson.js`)
- Integrates Leaflet maps with Vue reactivity
- Manages Australian region visualization
- Emits region selection events
- Handles map interactions and styling

### Sidebar Navigation (`sidebar.js`)
- Provides application navigation
- Displays LUTO branding
- Routes to different analysis views
- Responsive design for different screen sizes

## Setup and Usage

### Prerequisites
- A modern web browser with JavaScript enabled
- Python 3 (for local development server)

### Running Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/JinzhuWANG/VUE_LUTO.git
   cd VUE_LUTO
   ```

2. Start a local web server:
   ```bash
   python3 -m http.server 8000
   ```

3. Open your browser and navigate to:
   ```
   http://localhost:8000
   ```

### Data Transformation
If you need to update data from raw JSON files:

1. Place your JSON files in the `assets/` directory
2. Run the transformation script:
   ```bash
   python dataTransform/01_JSON2JS_dataTrans.py
   ```
3. This will convert JSON files to JavaScript files in the `data/` directory with the following features:
   - Properly formatted with indentation for better readability
   - Assigned to window objects with the same name as the source file
   - Map-related files prefixed with "map_" for clearer identification

### Production Deployment
The application is a static web application that can be deployed to any web server:
- Upload all files to your web server
- Ensure the web server can serve `.js` files with the correct MIME type
- No build process or server-side rendering required

## Model Integration

The dashboard is designed to work with LUTO 2.0 model outputs. Key integration points:

### Model Run Parameters
- **Version**: Model version tracking
- **Scenarios**: SSP (Shared Socioeconomic Pathways) and RCP (Representative Concentration Pathways)
- **Policy Settings**: Carbon pricing, biodiversity targets, agricultural management
- **Constraints**: Water usage, GHG emission limits, land use restrictions

### Data Outputs
The model generates datasets for:
- Economic indicators (revenue, costs)
- Land use areas by category
- GHG emissions by source
- Water usage by sector
- Biodiversity quality scores

## Development Notes

### Code Style
- Uses Vue 3 Composition API for component logic
- ES6+ JavaScript features
- No TypeScript (pure JavaScript implementation)
- Consistent naming conventions (camelCase for variables, kebab-case for components)

### Performance Considerations
- Locally hosted libraries for offline use and faster loading
- Lazy loading of data files to reduce initial load time
- Chart reuse and proper cleanup to prevent memory leaks
- Efficient map rendering with minimal DOM manipulation
- Responsive design principles for various screen sizes
- Structured data organization by region for faster access

### Browser Support
- Modern browsers supporting ES6+
- Vue 3 compatibility requirements
- WebGL support recommended for optimal map performance

## Contributing

When contributing to this project:
1. Maintain the existing code style and architecture
2. Test across different browsers and screen sizes
3. Ensure new data files follow the existing naming conventions
4. Update documentation for any new features or changes

## License

This project is part of the LUTO (Land Use Trade-Offs) model system. Please refer to the main LUTO project for licensing information.

## Version History

### Latest Changes
- Refactored data structure with improved JS file formatting
- Updated JSON to JS conversion process with better indentation
- Updated map UI components and views for improved visualization
- Enhanced map integration with dynamic data loading capabilities
- Optimized map data loading and UI controls for better performance
- Improved responsive UI with better layout and spacing
- Added Production analysis view for agricultural production data
- Implemented filterable dropdown components for better data selection
- Migrated from CDN dependencies to local libraries for offline use

### Future Enhancements
- Add more detailed analysis views for each domain
- Implement scenario comparison functionality
- Add data download options for raw model outputs
- Improve accessibility features for all visualizations
