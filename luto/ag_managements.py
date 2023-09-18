AG_MANAGEMENTS_INDEXING = {
    1: "Asparagopsis taxiformis",
}

AG_MANAGEMENTS_TO_LAND_USES = {
    "Asparagopsis taxiformis": [
        "Beef - natural land",
        "Beef - modified land",
        "Sheep - natural land",
        "Sheep - modified land",
        "Dairy - natural land",
        "Dairy - modified land",
    ]
}

# List of all agricultural managements, sorted by index value defined above
SORTED_AG_MANAGEMENTS = [item[1] for item in sorted(AG_MANAGEMENTS_INDEXING.items(), key=lambda item: item[0])]