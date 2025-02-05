import re
from gradio_client import Client, handle_file

class MolmoObjectLocator:
    def __init__(self, model_name="akhaliq/Molmo-7B-D-0924", api_name="/chatbot"):
        """
        Initialize the API client with the specified model and API endpoint.
        """
        self.client = Client(model_name)
        self.api_name = api_name

    def get_coordinates(self, image_path: str, object_name: str):
        """
        Given an image path and an object name, returns the (x, y) coordinates of the object.

        :param image_path: Path to the image file.
        :param object_name: Object to locate in the image.
        :return: Tuple (x, y) coordinates if found, else None.
        """
        # Define the text prompt
        query = f"Point to the {object_name}."

        # Call the chatbot API
        result = self.client.predict(
            image=handle_file(image_path),
            text=query,
            api_name=self.api_name
        )

        # Extract coordinates from response
        return self._extract_coordinates(result)

    def _extract_coordinates(self, response):
        """
        Private method to extract x, y coordinates from the API response.

        :param response: API response containing the object location.
        :return: Tuple (x, y) if coordinates are found, else None.
        """
        if isinstance(response, list) and len(response) > 0:
            response_text = response[0][1]  # Extract the relevant string containing coordinates
            match = re.search(r'point x="([\d.]+)" y="([\d.]+)"', response_text)
            if match:
                x, y = float(match.group(1)), float(match.group(2))
                return x, y
        return None 
