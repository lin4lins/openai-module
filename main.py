import os
from dotenv import load_dotenv
import openai

GENERATE_TAGS_PROMPT_TEMPLATE = "Give me {number_of_tags} tags (key words) from this text: {text}. Return tags only."

class OpenAIClient:
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        """
        Initializes the OpenAI client with the provided API key and model.

        :param client: The OpenAI client instance used to interact with the OpenAI API.
        :param model: The model to use for the request, default is 'gpt-3.5-turbo'
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def make_request(self, prompt: str) -> str:
        """
        Makes a request to the OpenAI API and returns the response content.

        :param prompt: The input prompt to send to the API
        :return: The parsed content from the response
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()

    def generate_tags(self, text: str, number_of_tags: int) -> list:
        """
        Generates a specified number of tags (keywords) from the input text
        and returns them as a list using the make_request() method.

        :param text: The input text to analyze
        :param number_of_tags: The number of tags to extract
        :return: A list of extracted tags
        """
        prompt = GENERATE_TAGS_PROMPT_TEMPLATE.format(number_of_tags=number_of_tags, text=text)
        parsed_content = self.make_request(prompt)
        return [tag.strip() for tag in parsed_content.split(",")]


def main():
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    client = OpenAIClient(api_key)

    file_path = "itzy.txt"
    with open(file_path, "r") as file:
        text = file.read()

    number_of_tags = 5
    result = client.generate_tags(text, number_of_tags)
    print(f"Extracted Tags: {result}")

if __name__ == "__main__":
    main()
