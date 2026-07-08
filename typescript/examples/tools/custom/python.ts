import "dotenv/config";
import { CustomTool } from "beeai-framework/tools/custom";

const customTool = await CustomTool.fromSourceCode(
  {
    // Ensure the env exists
    url: process.env.CODE_INTERPRETER_URL!,
    env: { API_URL: "https://riddles-api.vercel.app/random" },
  },
  `import requests
import os
from typing import Optional, Union, Dict

def get_riddle() -> Optional[Dict[str, str]]:
  """
  Fetches a random riddle from the Riddles API.

  This function retrieves a random riddle and its answer. It does not accept any input parameters.

  Returns:
      Optional[Dict[str, str]]: A dictionary containing:
          - 'riddle' (str): The riddle question.
          - 'answer' (str): The answer to the riddle.
      Returns None if the request fails.
  """
  url = os.environ.get('API_URL')
  
  try:
      response = requests.get(url)
      response.raise_for_status() 
      return response.json() 
  except Exception as e:
      return None`,
);
