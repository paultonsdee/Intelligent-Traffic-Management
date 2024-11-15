import re
import json
from config.config_logger import logger

def parse_gemini_response(response_text, w, h):
    """
    Parses the Gemini API response to extract ambulance bounding box data.
    """
    try:
        # Extract JSON code block using regex
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)

        if not json_match:
            logger.info("No JSON found in Gemini response.")
            return None

        json_str = json_match.group(0)
        data = json.loads(json_str)
        if "ambulance" in data:
            if data["ambulance"] == [] or data["ambulance"][0] == data["ambulance"][1] == data["ambulance"][2] == data["ambulance"][3]:
                logger.info("No ambulance detected")
                return None
            # Normalize bounding box coordinates
            data["ambulance"][0] /= w
            data["ambulance"][1] /= h
            data["ambulance"][2] /= w
            data["ambulance"][3] /= h
            return data["ambulance"]
        else:
            logger.info("No ambulance detected")
            return None
    except Exception as e:
        logger.warning(f"No ambulance detected or error while parsing the response: {e}")
    return None