from molmo import MolmoObjectLocator

# Initialize the API client
locator = MolmoObjectLocator()

# Specify image path and object name
image_path = "/Users/rheamalhotra/Desktop/teapot.jpg"
object_name = "teapot"

# Get coordinates
coordinates = locator.get_coordinates(image_path, object_name)

# Print result
if coordinates:
    print(f"Extracted Coordinates: x={coordinates[0]}, y={coordinates[1]}")
else:
    print("Could not extract coordinates from the response.")
