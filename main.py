import os
from dotenv import load_dotenv
import json

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


from prompts import vocab_search_prompt
from test_texts import user_info, vocab_test1, vocab_test2, phrase_test1, phrase_test2

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")


def simple_prompt_model_parser_chain(prompt_message, prompt_variables):
    gpt3_5 = ChatOpenAI(
        model="gpt-3.5-turbo",
        openai_api_key=openai_api_key,
    )
    prompt = ChatPromptTemplate.from_messages([("user", prompt_message)])
    output_parser = StrOutputParser()

    chain = prompt | gpt3_5 | output_parser

    result = json.loads(chain.invoke(prompt_variables))

    return result


test_prompt_variables1 = {
    "word": vocab_test1["word"],
    "context": vocab_test1["context"],
    "user_info": user_info,
}
test_prompt_variables2 = {
    "word": vocab_test2["word"],
    "context": vocab_test2["context"],
    "user_info": user_info,
}
test_prompt_variables3 = {
    "word": phrase_test1["word"],
    "context": phrase_test1["context"],
    "user_info": user_info,
}
test_prompt_variables4 = {
    "word": phrase_test2["word"],
    "context": phrase_test2["context"],
    "user_info": user_info,
}

result = simple_prompt_model_parser_chain(vocab_search_prompt, test_prompt_variables1)
print(f"\n{test_prompt_variables1}")
print(f"Definition: {result['definition']}")
print(f"Example: {result['example sentence']}")

result2 = simple_prompt_model_parser_chain(vocab_search_prompt, test_prompt_variables2)
print(f"\n{test_prompt_variables2}")
print(f"Definition: {result2['definition']}")
print(f"Example: {result2['example sentence']}")

result3 = simple_prompt_model_parser_chain(vocab_search_prompt, test_prompt_variables3)
print(f"\n{test_prompt_variables3}")
print(f"Definition: {result3['definition']}")
print(f"Example: {result3['example sentence']}")

result4 = simple_prompt_model_parser_chain(vocab_search_prompt, test_prompt_variables4)
print(f"\n{test_prompt_variables4}")
print(f"Definition: {result4['definition']}")
print(f"Example: {result4['example sentence']}")
