# VUE_LUTO - Land Use Trade-Offs (LUTO) 2.0 Dashboard

## Overview

VUE_LUTO is a web-based dashboard application for visualizing and analyzing results from the Land Use Trade-Offs (LUTO) 2.0 model. It provides interactive charts, maps, and data exploration tools for understanding the environmental, economic, and social impacts of different land use scenarios in Australia.

## Purpose

The LUTO model is designed to analyze trade-offs between different land uses in Australia, considering factors such as:
- **Economics**: Revenue, costs, and economic indicators across agricultural and non-agricultural sectors
- **Area Analysis**: Land use distribution and changes over time
- **Greenhouse Gas (GHG) Emissions**: Carbon footprint and climate impacts by land use type
- **Water Usage**: Water consumption and management across regions
- **Production**: Agricultural production and commodity analysis
- **Biodiversity**: Environmental conservation metrics including GBF (Global Biodiversity Framework) targets
- **Decision Variables (DVAR)**: Spatial optimization results and land use allocation

This dashboard provides an intuitive interface to explore model outputs through a progressive selection pattern, allowing users to drill down from national overviews to specific regions, land use types, and management practices.

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
│   ├── Biodiversity.js                 # Biodiversity analysis view
│   ├── Map.js                          # Interactive map view
│   ├── Settings.js                     # Application settings view
│   ├── Test.js                         # Test view for development
│   └── NotFound.js                     # 404 error page
├── services/                           # Service modules
│   ├── ChartService.js                  # Data handling service
│   └── MapService.js                   # Map data and interactions service
├── routes/                             # Routing configuration
│   └── route.js                        # Vue Router setup
├── data/                               # Data files and model outputs
│   ├── chart_option/                   # Chart configuration templates
│   │   ├── Chart_default_options.js    # Default chart styles
│   │   └── chartMemLogOptions.js       # Memory log chart configuration
│   ├── geo/                            # Geographic data (Australian regions)
│   ├── map_layers/                     # Map visualization data
│   │   ├── map_area_*.js               # Area map data files
│   │   ├── map_bio_*.js                # Biodiversity map data files
│   │   ├── map_cost_*.js               # Cost map data files
│   │   ├── map_revenue_*.js            # Revenue map data files
│   │   ├── map_GHG_*.js                # GHG emissions map data files
│   │   ├── map_water_*.js              # Water usage map data files
│   │   ├── map_quantities_*.js         # Production quantities map data files
│   │   └── map_dvar_*.js               # Decision variables map data files
│   ├── Area_*.js                       # Area analysis chart data
│   ├── Economics_*.js                  # Economics analysis chart data
│   ├── GHG_*.js                        # GHG emissions chart data
│   ├── Water_*.js                      # Water usage chart data
│   ├── Production_*.js                 # Production analysis chart data
│   ├── BIO_*.js                        # Biodiversity analysis chart data
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

### 2. Analysis Views
Each analysis view follows a progressive selection pattern for data exploration:

#### Area Analysis
- Land use area distribution across agricultural (Ag), agricultural management (Ag Mgt), and non-agricultural (Non-Ag) categories
- Temporal analysis of land use changes over simulation years
- Progressive selection: Region → Category → Water → Landuse

#### Economics Analysis
- Revenue and cost analysis with dual chart/map visualization
- Separate cost and revenue map layers with different data structures
- Combined cost/revenue chart data with validation for agricultural management categories
- Progressive selection with special handling for cost/revenue switching

#### GHG Emissions Analysis
- Greenhouse gas emissions by land use type and management practice
- Comparison across agricultural and non-agricultural sectors
- Progressive selection: Region → Category → AgMgt → Water → Landuse

#### Water Usage Analysis
- Water yield and consumption analysis by Natural Resource Management (NRM) regions
- Agricultural vs non-agricultural water usage patterns
- Progressive selection with NRM-specific data structures

#### Production Analysis
- Agricultural commodity production quantities and trends
- Production targets and achievement analysis
- Export, import, and domestic consumption breakdowns
- Progressive selection: Region → Category → Water → Landuse

#### Biodiversity Analysis
- Global Biodiversity Framework (GBF) target analysis including GBF2, GBF3, GBF4, GBF8
- Biodiversity quality metrics and species conservation indicators
- Ecological and species-level biodiversity assessments
- Simplified progressive selection: Region → [series] for most datasets

#### Map View (Decision Variables)
- Spatial visualization of optimization results
- Land use allocation and water supply decisions
- Agricultural management practice distribution
- Simplified hierarchy: Category → Landuse/AgMgt → Year

### 3. Interactive Map System
- **Australian Regions**: Based on Natural Resource Management (NRM) regions
- **Hover Effects**: Region highlighting and tooltips
- **Region Selection**: Click to select regions for detailed analysis
- **Layer Visualization**: Supports multiple map layers including area, cost, revenue, GHG, water, biodiversity, and decision variables
- **Base64 Image Rendering**: Efficient map tile rendering with bounds and min/max value information
- **Responsive Design**: Adapts to different screen sizes

### 4. Chart System
- **Highcharts Integration**: Professional-grade interactive charts
- **Export Capabilities**: PNG, JPEG, PDF, and CSV export options
- **Accessibility**: Screen reader support and keyboard navigation
- **Responsive Design**: Charts adapt to container sizes
- **Dual Series Support**: Special handling for combined cost/revenue series in economics module

### 5. Progressive Selection Pattern
All analysis views implement a standardized progressive selection architecture:
- **Cascade Watchers**: Automatic downstream option updates when upstream selections change
- **Memory Preservation**: Previous selections restored when switching between categories
- **Data Validation**: Safe property access with optional chaining and fallback values
- **Hierarchical Data Access**: Structured data access patterns based on module type

## Data Architecture

### Dynamic Data Loading
The application uses a custom script loading system (`helpers.js`) that:
- Loads data files on-demand to optimize performance
- Manages script dependencies and loading order
- Provides error handling for failed data loads
- Supports timeout mechanisms for reliable loading

### Data Types and Structure

#### Core Data Categories
1. **Supporting Info** (`Supporting_info.js`): Consolidated model run settings and metadata
2. **Chart Data**: Hierarchical time-series data organized by progressive selection patterns:
   - **Standard Full**: Category → AgMgt → Water → Landuse → [series]
   - **Standard Simple**: Category → Water → Landuse → [series]
   - **NonAg Simplified**: Category → Landuse → [series] (no Water/AgMgt levels)
   - **Economics Special**: Dual cost/revenue series in single files with validation
3. **Map Data**: Spatial visualization data ending with `{img_str: "base64...", bounds: [...], min_max: [...]}`
4. **Geographic Data** (`geo/NRM_AUS.js`): GeoJSON data for Australian NRM regions

#### Data File Patterns
- **Chart Files**: `ModuleName_Category.js` (e.g., `Area_Ag.js`, `BIO_GBF2_NonAg.js`)
- **Map Files**: `map_type_Category.js` (e.g., `map_area_Ag.js`, `map_cost_Am.js`)
- **Overview Files**: `ModuleName_overview_*.js` for summary visualizations
- **Ranking Files**: `ModuleName_ranking.js` for comparative analysis

#### Progressive Selection Hierarchies
- **Water Level Options**: `"ALL"`, `"Dryland"`, `"Irrigated"` (Ag/AgMgt only)
- **AgMgt Options**: `"ALL"`, `"AgTech EI"`, `"Asparagopsis taxiformis"`, `"Biochar"`, `"Precision Agriculture"`
- **Category Types**: `"Ag"` (Agricultural), `"Ag Mgt"` (Agricultural Management), `"Non-Ag"` (Non-Agricultural)

#### Special Data Structures
- **Economics Module**: Separate cost/revenue map files but combined chart data with mixed series arrays
- **Biodiversity Module**: Multiple GBF target datasets (GBF2, GBF3, GBF4, GBF8) plus quality metrics
- **DVAR Module**: Simplified map-only structure with direct category access

## Component Architecture

### Chart Container (`chart_container.js`)
- Wraps Highcharts functionality in a Vue component
- Manages chart lifecycle (creation, updates, destruction)
- Handles loading states and error conditions
- Supports reactive data updates and dual series rendering

### Map Components
- **Map GeoJSON** (`map_geojson.js`): Leaflet integration with Vue reactivity for region visualization
- **Regions Map** (`regions_map.js`): Interactive regional selection component
- Manages Australian NRM region visualization with base64 image overlay support
- Emits region selection events and handles map interactions

### UI Components
- **Sidebar Navigation** (`sidebar.js`): Application navigation with LUTO branding and responsive design
- **Filterable Dropdown** (`filterable_dropdown.js`): Searchable dropdown component for data selection
- **Ranking Cards** (`ranking_cards.js`): Comparative ranking visualization component

### Service Layer
- **ChartService** (`services/ChartService.js`): Chart data registration and management
- **MapService** (`services/MapService.js`): Map layer data registration and spatial data handling
- **Helpers** (`components/helpers.js`): Utility functions for dynamic script loading and data management

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
The model generates comprehensive datasets organized by module:

#### Area Module
- Land use area distribution across Ag, Ag Mgt, and Non-Ag categories
- Temporal changes in land allocation over simulation years
- Regional breakdown by NRM areas

#### Economics Module
- Revenue and cost analysis with separate map layers but combined chart data
- Economic indicators across agricultural and non-agricultural sectors
- Cost-benefit analysis by land use type and management practice

#### GHG Module
- Greenhouse gas emissions by land use category and management type
- Carbon footprint analysis across agricultural and non-agricultural sectors
- Temporal emissions trends and mitigation potential

#### Water Module
- Water yield and consumption by NRM region
- Agricultural vs non-agricultural water usage patterns
- Water management efficiency metrics

#### Production Module
- Agricultural commodity production quantities and trends
- Export, import, and domestic consumption analysis
- Production target achievement metrics

#### Biodiversity Module
- Global Biodiversity Framework (GBF) target indicators (GBF2, GBF3, GBF4, GBF8)
- Species conservation metrics and habitat quality assessments
- Ecological connectivity and biodiversity quality scores

#### Decision Variables (DVAR) Module
- Spatial optimization results showing optimal land use allocation
- Agricultural management practice distribution
- Water supply and infrastructure decisions

## Development Notes

### Code Style
- Uses Vue 3 Composition API with reactive programming patterns
- ES6+ JavaScript features with consistent coding standards
- No TypeScript (pure JavaScript implementation)
- Consistent naming conventions (camelCase for variables, kebab-case for components)
- Progressive selection pattern implementation across all analysis views

### Progressive Selection Architecture
All analysis views implement standardized cascade watcher patterns:
- **Memory Preservation**: Previous selections restored when switching categories
- **Cascading Updates**: Downstream options automatically update when upstream selections change
- **Data Validation**: Safe property access with optional chaining and fallback values
- **No Manual Clearing**: Automated handling eliminates need for manual array/selection clearing

### Performance Considerations
- Locally hosted libraries for offline use and faster loading
- On-demand data loading with script management system
- Chart reuse and proper cleanup to prevent memory leaks
- Efficient map rendering with base64 image tiles and minimal DOM manipulation
- Responsive design principles for various screen sizes
- Hierarchical data organization optimized for progressive selection patterns
- Memory-efficient cascade watchers that preserve user experience

### Browser Support
- Modern browsers supporting ES6+
- Vue 3 compatibility requirements
- WebGL support recommended for optimal map performance

## Contributing

When contributing to this project:
1. **Architecture Consistency**: Follow the progressive selection pattern for all analysis views
2. **Code Standards**: Maintain existing Vue 3 Composition API patterns and cascade watcher implementations
3. **Data Structure**: Ensure new data files follow established naming conventions and hierarchical patterns
4. **Testing**: Test across different browsers, screen sizes, and data selection scenarios
5. **Documentation**: Update documentation for any new features, data modules, or architectural changes
6. **Progressive Selection**: Implement standardized cascade patterns from existing views (e.g., Area.js) when creating new analysis modules

## License

This project is part of the LUTO (Land Use Trade-Offs) model system. Please refer to the main LUTO project for licensing information.

## Version History

### Latest Changes
- **Progressive Selection Architecture**: Implemented standardized cascade watcher patterns across all analysis views
- **Biodiversity Module**: Added comprehensive GBF (Global Biodiversity Framework) target analysis including GBF2, GBF3, GBF4, GBF8
- **Map System Enhancement**: Added Map view (Decision Variables) with spatial optimization results visualization
- **Data Structure Expansion**: Enhanced data organization with 80+ chart files and comprehensive map layer support
- **Economics Module Specialization**: Implemented dual cost/revenue visualization with separate map layers and combined chart data
- **Memory Preservation**: Added intelligent selection memory across category switching for improved user experience
- **Service Layer**: Separated ChartService and MapService for better data management and registration
- **UI Component Expansion**: Added filterable dropdowns and ranking cards for enhanced data interaction

### Architecture Highlights
- **7 Main Analysis Views**: Area, Economics, GHG, Water, Production, Biodiversity, Map (DVAR)
- **Progressive Selection Patterns**: Standardized data navigation across all modules with 3-4 hierarchy levels
- **80+ Data Files**: Comprehensive chart and map data coverage across all analysis domains
- **Dual Visualization**: Chart and map integration with synchronized data selection
- **Memory-Efficient Watchers**: Automated cascade patterns that preserve user selections and eliminate manual clearing

### Future Enhancements
- Enhanced scenario comparison functionality across multiple model runs
- Advanced data export capabilities for custom analysis
- Improved accessibility features with enhanced keyboard navigation
- Real-time collaboration features for multi-user analysis sessions
