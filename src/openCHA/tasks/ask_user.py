from typing import Any
from typing import List

from openCHA.tasks import BaseTask


class AskUser(BaseTask):
    """
    **Description:**

        This task is asking question back to the user and stops planning. When needed, the planner will decide to ask question from user
        and use the user's answer to proceed to the planning.

    """

    name: str = "ask_user"
    chat_name: str = "AskUser"
    description: str = "Engages with the user for additional information or direct responses."
    dependencies: List[str] = []
    inputs: List[str] = [
        "The text returned to user. It should be relevant and very detailed based on the latest user's Question."
    ]
    outputs: List[str] = ["Returns a string containing the user response to the question. For example: 'I don't like cats'"]
    output_type: bool = False
    return_direct: bool = True

    def _execute(
        self,
        inputs: List[Any] = None,
    ) -> str:
        """Translate query"""
        if inputs is None:
            return ""
        return input(inputs[0])

    def explain(
        self,
    ) -> str:
        return "This task simply asks user to provide more information or continue interaction."
