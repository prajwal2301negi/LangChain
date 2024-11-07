# blip-image-captioning-large
# library using for captioning by salesforce

# detr-resnet-50
# by facebook to detect object

# will be creating web application in main.py, functions in function.py to detect object and caption image, and use langchain for ImageCaptionTool and ObjectDetectionTool.


# then we will initialize the agent in main.py before webapplication and coumpute agent response after webapplication.

#streamlit is use for creating web application


import streamlit as st


# INITIALIZE AGENT
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from tools import ImageCaptionTool, ObjectDetectionTool

tools = [ImageCaptionTool(), ObjectDetectionTool()]

# As we want the agent to remember previos responses
conversational_memory = ConversationBufferWindowMemory(
    memory_key = 'chat_history',
    # length/size of agent memory
    k = 5,
    return_messages = True,
)

# As we communicate with agent, we need Large Language Model.
llm = ChatOpenAI(
    openai_api_key = 'YOUR_API_KEY',
    # output of agent will be more exact
    temperature = 0,
    model_name = 'gpt-3.5-turbo'
)

# Initialize agent and input types.
agent = initialize_agent(
    agent = "chat-conversational-react-description",
    # Providing agent with tools and llm
    tools = tools,
    llm = llm, 
    max_iterations = 5,
    # As we want to see/ aware of exactly what the agent is doing 
    verbose = True,
    memory = conversational_memory,
    early_stopping_memory = 'generate'
)



# Title
st.title('Ask a question for the image')

#Header
st.header('Please upload an image')

#Upload Image
file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if file:
    # Display Image
    st.image(file, caption='Uploaded Image',
             use_column_width=True)
    
    # Text Input
    user_question = st.text_input('Ask a qustion about imamge:')



    # COMPUTE AGENT RESPONSE
    from tempfile import NamedTemporaryFile # for image_path

    with NamedTemporaryFile(dir = '.') as f:
        f.write(file.getbuffer())
        image_path = f.name
        # storing the image in temp file with name

        if user_question and user_question != "":
            with st.spinner(text="In progress..."):
                response = agent.run('{}, this is the image path: {}'.format(user_question, image_path))
                
                # Write agent response
                st.write(response)