from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

class Generator:
    def __init__(self, model):
        self.model = model
        
    def format_prompt(self, question):
        system_prompt = open("prompts/system_prompt.txt", "r").read()
        user_prompt = open("prompts/user_prompt.txt", "r").read()
        user_prompt = user_prompt.format(question)
        template = "{} \n {}".format(system_prompt, user_prompt)  
        return template
    
    def generate_response(self, question):
        template = self.format_prompt(question) 
        prompt = ChatPromptTemplate.from_template(template)
        return prompt