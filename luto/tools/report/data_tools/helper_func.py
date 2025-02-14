# Copyright 2025 Bryan, B.A., Williams, N., Archibald, C.L., de Haan, F., Wang, J., 
# van Schoten, N., Hadjikakou, M., Sanson, J.,  Zyngier, R., Marcos-Martinez, R.,  
# Navarro, J.,  Gao, L., Aghighi, H., Armstrong, T., Bohl, H., Jaffe, P., Khan, M.S., 
# Moallemi, E.A., Nazari, A., Pan, X., Steyl, D., and Thiruvady, D.R.
#
# This file is part of LUTO2 - Version 2 of the Australian Land-Use Trade-Offs model
#
# LUTO2 is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# LUTO2 is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# LUTO2. If not, see <https://www.gnu.org/licenses/>.

import os
from lxml import etree
from  lxml.etree import Element



def list_all_files(directory):

    """
    This function is used to get all the file paths under the given directory

    Parameters:
    directory (str): The given directory.

    Returns:
    list: The list of file paths under the given directory.
    """
    
    file_list = []
    for root, dirs, files in os.walk(directory):
        file_list.extend(os.path.join(root, file) for file in files)
    return file_list
    

def add_txt_2_html(html_path:str, txt:str, id:str)->None:
    """
    Adds text content to an HTML file at a specified location.

    Args:
        html_path (str): The file path of the HTML file.
        txt (str): The text content to be added to the HTML file.
            - It can be either a string or a file path.
        id (str): The ID of the HTML element where the text content should be added.

    Returns:
        None
    """
    
    # Get the tree of the index page
    index_tree = etree.parse(html_path, etree.HTMLParser())
    
    # Check if txt is a file path
    if os.path.exists(os.path.abspath(txt)):
        # If it is, read the file
        with open(txt,'r') as f:
            txt = f.read()
    
    # Add a new div element to the page      
    data_csv_div = index_tree.find('.//div[@id="data_csv"]')

    pre_element = etree.SubElement(data_csv_div, "pre")
    pre_element.set("id", id)
    pre_element.text = txt
    
    # Write changes to the html
    index_tree.write(html_path, 
                    pretty_print=True,
                    encoding='utf-8',
                    method='html')
        
    
    

def add_data_2_html(html_path:str, data_pathes:list)->None:
    """
    Adds data from multiple files to an HTML file.

    Args:
        html_path (str): The path to the HTML file.
        data_pathes (list): A list of paths to the data files.

    Returns:
        None: This function does not return anything.
    """

    # Step 1: Parse the HTML file
    parser = etree.HTMLParser()
    tree = etree.parse(html_path, parser)
    name = os.path.basename(html_path)

    # Step 2: Remove the data_csv if it exists
    data_csv_div = tree.find('.//div[@id="data_csv"]')

    if data_csv_div is not None:
        data_csv_div.getparent().remove(data_csv_div)

    # Step 3: Append a div[id="data_csv"] to the div[class="content"]
    content_div = tree.xpath('.//div[@class="content"]')[0]

    new_div = Element("div",)
    new_div.set("id", "data_csv")
    new_div.set("style", "display: none;")

    # Step 4: Create and append five <pre> elements
    for data_path in data_pathes:

        # get the base name of the file
        data_name = os.path.basename(data_path).split('.')[0]

        with open(data_path, 'r') as file:
            raw_string = file.read()

        pre_element = etree.SubElement(new_div, "pre")
        pre_element.set("id", f"{data_name}_csv")
        pre_element.text = raw_string


    # Step 5: Insert the new div
    content_div.addnext(new_div)

    # Step 6: Save the changes
    tree.write(html_path, method="html")

    print(f"Data of {name} has successfully added to HTML!")
    
    
    
def select_years(year_list):
    """
    Selects a subset of years from the given year_list. The selected years will 
    1) include the first and last years in the year_list, and
    2) be evenly distributed between the first and last years.

    Args:
        year_list (list): A list of years.

    Returns:
        list: A list containing the selected years.

    """
    year_list = sorted(year_list)

    if year_list[-1] == 2050:
        # Return [2010, 2020, ..., 2050] if the last year is 2050 
        return list(range(2010, 2051, 10))
    elif len(year_list) <= 6:
        # Return the list as is if 5 or fewer elements
        return year_list
    else:
        # Select the start and end years
        selected_years = [year_list[0], year_list[-1]]

        # Calculate the number of years to be selected in between (4 in this case)
        slots = 4

        # Calculate the interval for selection
        interval = (len(year_list) - 2) / (slots + 1)

        # Select years based on calculated interval, ensuring even distribution
        selected_years.extend(
            year_list[int(round(i * interval))] for i in range(1, slots + 1)
        )
        return selected_years