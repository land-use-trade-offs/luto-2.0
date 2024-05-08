from luto.tools.create_task_runs.helpers import create_grid_search_template, create_task_runs, create_settings_template



# Create a template for the custom settings, and then create the custom settings
create_settings_template()


# Create a template for the grid search, and then create the grid search
create_grid_search_template(num_runs=20)


# Create the task runs
create_task_runs()



