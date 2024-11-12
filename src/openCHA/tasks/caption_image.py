from typing import Any
from typing import List

from openCHA.tasks import BaseTask
from openCHA.datapipes.datapipe import DataPipe

import os
from openai import OpenAI
import base64

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


class CaptionImage(BaseTask):
    name: str = "caption_image"
    chat_name: str = "CaptionImage"
    description: str = "analyzes the image and returns image content as text description."
    dependencies: List[str] = []
    inputs: List[str] = [
              "metadata file key",
              "sub-query of interest, e.g. 'identify a food item in the image'",
              "metadata file name"
    ]
    datapipe: DataPipe = None
    outputs: List[str] = [
        "Returns a string with answer for image sub-query, for example: 'image shows a carrot cake'"
    ]
    output_type: bool = False
    return_direct: bool = True

    def parse_input(
        self,
        input: str,
    ) -> List[str]:
        """
        Parse the input string into a list of strings.

        Args:
            input (str): Input string to be parsed.
        Return:
            List[str]: List of parsed strings.

        """
 
        return input.split("$#")

    def _execute(
        self,
        inputs: List[str] = None,
    ) -> str:
        """
        Abstract method representing the execution of the task.

        Args:
            input (str): Input data for the task.
        Return:
            str: Result of the task execution.
        Raise:
            NotImplementedError: Subclasses must implement the execute method.


        """

        if not api_key:
            print("API Key not found. Make sure OPENAI_API_KEY is set in the environment.")
     
        # TODO error check
        image_path = inputs[0]        
        query = inputs[1]

        base64_image = encode_image(image_path)
        response = client.chat.completions.create(
           model="gpt-4o-mini",
           messages=[
              {
                "role": "user",
                "content": [
                     {
                       "type": "text",
                       "text": "What is in this image?",
                     },
                     {
                       "type": "image_url",
                       "image_url": {
                          "url":  f"data:image/jpeg;base64,{base64_image}"
                       },
                     },
                ],
              }
           ],
        )

        description = response.choices[0].message.content
        print(description)
        return description

    def explain(
        self,
    ) -> str:
        """
        Provide a sample explanation for the task.

        Return:
            str: Sample explanation for the task.

        """

        return "This task returns a detailed image description."
