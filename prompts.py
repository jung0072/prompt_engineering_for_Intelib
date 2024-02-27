vocab_search_prompt = """"
What does {word} mean in context of {context}? 

Reply in the following json format:
    "definition" : 
    "example sentence" : 

For the example sentence, create one that is relevant to the user's background. User's background = {user_info}
"""
