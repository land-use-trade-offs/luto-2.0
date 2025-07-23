// Create Chart
document.addEventListener("DOMContentLoaded", function () {

    const support_info = JSON.parse(document.getElementById('Supporting_info').innerText);
    const colors = support_info.colors;
    const model_years = support_info.years;


    // Get the available years for plotting
    var years = model_years.map(function (x) { return parseInt(x); });
    years.sort(function (a, b) { return a - b; });


    // Set the title alignment to left
    Highcharts.setOptions({
        colors: colors,
        title: {
            align: 'left'
        }
    });
    
    // Define title mapping for different biodiversity types
    const bioTypeTitles = {
        'BIO_quality': 'Biodiversity Quality',
        'BIO_GBF2': 'GBF Target 2',
        'BIO_GBF3': 'GBF Target 3 - National Vegetation Information System (NVIS)',
        'BIO_GBF4_ECNES': 'GBF Target 4 - Species of Ecological Communities of National Environmental Significance (ECNES)',
        'BIO_GBF4_SNES': 'GBF Target 4 - Species of National Environmental Significance (SNES)',
        'BIO_GBF8_GROUP': 'GBF Target 8 - Species Scores by Group',
        'BIO_GBF8_SPECIES': 'GBF Target 8 - Species Scores by Species'
    };
    
    // Define metric view mapping for readable names
    const metricViewTitles = {
        'overview_1_Type': 'Overview',
        'split_Ag_1_Landuse': 'Agricultural Land-use',
        'split_Am_2_Agri-Management': 'Agricultural Management',
        'split_Am_1_Landuse': 'Agricultural Management by Land-use',
        'split_NonAg_1_Landuse': 'Non-Agricultural Land-use'
    };
    
    
    // Chart configuration templates
    const chartConfigs = {
        // Template for basic chart configuration
        baseConfig: {
            chart: {
                type: "column",
                marginRight: 380,
            },
            yAxis: {
                title: {
                    text: "Contribution percent (2010=100%)",
                },
            },
            plotOptions: {
                column: {
                    stacking: "normal",
                },
            },
            tooltip: {
                formatter: function () {
                    return `<b>Year:</b> ${this.x}<br><b>${this.series.name
                        }:</b>${this.y.toFixed(2)}<br/>`;
                },
            },
            legend: {
                align: "right",
                layout: "vertical",
                x: -150,
                verticalAlign: "middle",
            },
            credits: {
                enabled: false,
            },
        },
        
        // Custom y-axis titles for different biodiversity types
        yAxisTitles: {
            'BIO_GBF3': "Contribution percent (Pre-1750=100%)",
            'default': "Contribution percent (2010=100%)",
        },
        
        // Chart title templates
        titleTemplates: {
            'overview_1_Type': "{bioType} Overview",
            'split_Ag_1_Landuse': "{bioType} Agricultural Land-use Contribution",
            'split_Am_2_Agri-Management': "{bioType} Agricultural Management Contribution",
            'split_Am_1_Landuse': "{bioType} Agricultural Management by Land-use Contribution",
            'split_NonAg_1_Landuse': "{bioType} Non-Agricultural Land-use Contribution",
        }
    };
    
    // List of biodiversity types that should display as grid charts
    const gridChartTypes = ['BIO_GBF3', 'BIO_GBF4_ECNES', 'BIO_GBF4_SNES', 'BIO_GBF8_GROUP', 'BIO_GBF8_SPECIES'];
    
    // Function to create and show the selected chart
    function showSelectedChart() {
        const bioType = document.getElementById('bio_type').value;
        let metricsView = document.getElementById('metrics_view').value;
        
        // Update the page title
        document.getElementById('biodiversity_title').textContent = bioTypeTitles[bioType];
        
        // Clear the chart container
        const chartContainer = document.getElementById('chart_container');
        chartContainer.innerHTML = '';
        
        // Get the data element
        const dataElement = document.getElementById(bioType + '_' + metricsView);
        
        if (dataElement && dataElement.innerHTML) {
            try {
                // Parse the data
                const data = JSON.parse(dataElement.innerHTML).AUSTRALIA;
                
                // Check if this is a grid chart type (GBF3/4/8)
                const isGridChart = gridChartTypes.includes(bioType);
                
                if (isGridChart) {
                    // Grid of species charts
                    chartContainer.classList.add('block-container');
                    
                    // Create a chart for each species
                    Object.entries(data).forEach(([name, seriesData], index) => {
                        // Create a chart block for each species/group
                        const chartBlock = document.createElement('div');
                        chartBlock.classList.add('chart-block');
                        chartBlock.id = `grid-chart-${index}`;
                        chartContainer.appendChild(chartBlock);
                        
                        // Prepare chart configuration
                        const config = JSON.parse(JSON.stringify(chartConfigs.baseConfig)); // Deep clone
                        
                        // Set chart title to species/group name
                        config.title = { text: name, align: 'center' };
                        
                        // Set Y-axis title based on biodiversity type
                        const yAxisTitle = chartConfigs.yAxisTitles[bioType] || chartConfigs.yAxisTitles['default'];
                        config.yAxis.title.text = yAxisTitle;
                        
                        // Set data series
                        config.series = seriesData;
                        
                        // Smaller margin for grid charts
                        config.chart.marginRight = 100;
                        
                        // Adjust legend for grid charts
                        config.legend.align = 'center';
                        config.legend.layout = 'horizontal';
                        config.legend.x = 0;
                        config.legend.y = 20;
                        config.legend.verticalAlign = 'bottom';
                        
                        // Create the chart
                        Highcharts.chart(chartBlock.id, config);
                    });
                } else {
                    // Standard single chart view for non-grid types
                    chartContainer.classList.remove('block-container');
                    
                    // Create a div for the chart
                    const chartDiv = document.createElement('div');
                    chartDiv.className = 'fig';
                    chartDiv.id = 'dynamic-chart';
                    chartContainer.appendChild(chartDiv);
                    
                    // Prepare chart configuration
                    const config = JSON.parse(JSON.stringify(chartConfigs.baseConfig)); // Deep clone
                    
                    // Set chart title
                    const titleTemplate = chartConfigs.titleTemplates[metricsView] || "{bioType}";
                    const chartTitle = titleTemplate.replace("{bioType}", bioTypeTitles[bioType].replace(/ - .*$/, ''));
                    config.title = { text: chartTitle };
                    
                    // Set Y-axis title based on biodiversity type
                    const yAxisTitle = chartConfigs.yAxisTitles[bioType] || chartConfigs.yAxisTitles['default'];
                    config.yAxis.title.text = yAxisTitle;
                    
                    // Set data series
                    config.series = data;
                    
                    // Create the chart
                    Highcharts.chart('dynamic-chart', config);
                }
            } catch (error) {
                console.error('Error creating chart:', error);
                chartContainer.innerHTML = `<p>Error creating chart: ${error.message}</p>`;
            }
        } else {
            chartContainer.innerHTML = '<p>Chart data not available for this selection.</p>';
        }
    }
    
    // Adjust the metrics view options based on the selected biodiversity type
    function updateMetricsViewOptions() {
        const bioType = document.getElementById('bio_type').value;
        const metricsView = document.getElementById('metrics_view');
        
        // Save current selection if possible
        const currentSelection = metricsView.value;
        
        // Clear options
        metricsView.innerHTML = '';
        
        // Add common options for all types
        const commonOptions = ['overview_1_Type', 'split_Ag_1_Landuse', 'split_Am_2_Agri-Management', 'split_NonAg_1_Landuse'];
        
        // Add 'split_Am_1_Landuse' for specific bio types that support it
        const landUseOptions = [...commonOptions];
        if (bioType === 'BIO_quality' || bioType === 'BIO_GBF3' || 
            bioType === 'BIO_GBF8_GROUP' || bioType === 'BIO_GBF8_SPECIES') {
            landUseOptions.splice(3, 0, 'split_Am_1_Landuse');
        }
        
        // Add options to the select
        landUseOptions.forEach(option => {
            const optElement = document.createElement('option');
            optElement.value = option;
            optElement.textContent = metricViewTitles[option];
            metricsView.appendChild(optElement);
        });
        
        // Try to restore previous selection if it exists in new options
        if (landUseOptions.includes(currentSelection)) {
            metricsView.value = currentSelection;
        }
    }
    
    // Event listeners for the dropdown selects
    document.getElementById('bio_type').addEventListener('change', function() {
        updateMetricsViewOptions();
        showSelectedChart();
    });
    
    document.getElementById('metrics_view').addEventListener('change', function() {
        showSelectedChart();
    });
    
    
    // Initialize the selection interface
    updateMetricsViewOptions();
    showSelectedChart();

});
