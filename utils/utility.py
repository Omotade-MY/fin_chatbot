import streamlit as st
from langchain.chat_models import ChatOpenAI
import os
from langchain.prompts.prompt import PromptTemplate
import openai
import lida
from lida import Manager, TextGenerationConfig , llm  
import pandas as pd

class ExcelLoader():
    def __init__(self, file):
        import pandas as pd
        self.status = False
        self.name =  'ExcelLoader'
        self.file = file
        self.loader = pd.ExcelFile
        self.ext = ['xlsx']
    
    def load(self):
        from langchain.document_loaders.csv_loader import CSVLoader

        ssheet = self.loader(self.file)
        # try:
        #     os.mkdir('temp')

        # except FileExistsError:
        #     pass
        docs = []
        for i,sheet in enumerate(ssheet.sheet_names):
            #df = ssheet.parse(sheet)
            temp_path = f'{sheet}.csv'
            docs.append(temp_path)
            #df.to_csv(temp_path, index=False)
        return docs

def process_csv_file(file):
    file_paths = []
    if file.split('.')[-1] == 'csv':
        file_paths.append(file)
    elif file.split('.')[-1] == 'xlsx':
        loader = ExcelLoader(file)
        paths = loader.load()
        file_paths.extend(paths)
    if len(file_paths) == 1:
        return file_paths[0]
    return file_paths


def randomName():
    import numpy as np
    n = []
    n.append(np.random.choice(['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']).upper())
    n.append(str(np.random.randint(1,9)))                                 
    n.append(str(np.random.randint(1,9)))   
    n.append(np.random.choice(['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']))         
    n.append(np.random.choice(['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']).upper())  

    return ''.join(n)       



def generate_plot(data_path, prompt=None,api_key=None):

    assert data_path is not None
    directory_path = 'images'
    # Check if the directory already exists
    if not os.path.exists(directory_path):
        # Create the directory
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' was created.")
    else:
        print(f"Directory '{directory_path}' already exists.")
    
    lida = Manager(text_gen = llm(provider="openai", api_key=api_key)) 
    textgen_config = TextGenerationConfig(n=1, temperature=0.5, model="gpt-3.5-turbo-0301", use_cache=True)
    
    summary = lida.summarize(data_path, summary_method="default", textgen_config=textgen_config)  
    
    #textgen_config = TextGenerationConfig(n=1, temperature=0, use_cache=True)
    if prompt == None:
        goals = lida.goals(summary, n=1, textgen_config=textgen_config)

    else:
        persona = prompt
        goals = lida.goals(summary, n=1, persona=persona, textgen_config=textgen_config)
        
    i = 0
    library = "seaborn"
    plots = lida.visualize(summary=summary, goal=goals[i], textgen_config=textgen_config, library=library,)
    
    if len(plots) == 0:
        library = "matplotlib"
        textgen_config = TextGenerationConfig(n=1, temperature=0, use_cache=False)
        plots = lida.visualize(summary=summary, goal=goals[i], textgen_config=textgen_config, library=library,)

    if len(plots) == 0:
        st.write("Could not generate a plot from your prompt. The below chart can be helpful")
        caption, img_path = generate_plot(data_path=data_path)
        return caption, img_path
    
    
    fig = plots[0]
    
    
    #if len(plots) != 0:
    #fig = plots[0]

    img_name = randomName() + ".jpg"
    img_path  = os.path.join(os.getcwd(), directory_path, img_name)
    lida.edit(code=fig.code, summary=summary, instructions=f"Save the figure to the file '{img_path}'")
    caption = goals[i].rationale
    # st.write(goals[i].visualization)

    
    return caption, img_path


def classify_prompt(input, api_key=None):
    """
    This function classifies a user input to determine if it is requesting a plot.
    It returns True if the input is requesting a plot, and False otherwise.
    """

    # Updated prompt with clearer instructions
    prompt = """
    Analyze the following user input and determine if it is requesting a plot of a story, data, or any other form of graphical representation.
    
    If the input explicitly asks for a plot, description of a plot, or any graphical representation, respond with "PLOT REQUESTED".
    If the input does not ask for a plot or graphical representation, respond with "NO PLOT REQUESTED".
    
    User Input: "{}"
    """

    model = ChatOpenAI(model_name="gpt-3.5-turbo-16k-0613")
    res = model.predict(prompt.format(input))

    # Checking for the specific responses
    if res.strip() == "PLOT REQUESTED":
        return True
    else:
        return False



def display(img_path, rationale):
    from PIL import Image

    im = Image.open(img_path)
    container = st.container()
    with container:
        st.image(im)
        st.markdown("**Insight**")
        st.write(rationale)
        from io import BytesIO
        buf = BytesIO()
        im.save(buf, format="JPEG")
        byte_im = buf.getvalue()
        st.download_button(label="Download Image",
                           data=byte_im,
                           file_name=f"img{randomName()}.jpg",
                           mime="image/jpeg",)
        
def data_load(ext, file):
    try:
        os.mkdir('temps')
    except FileExistsError:
        pass
    path = os.path.join(os.getcwd(), r"temps")
    if ext == "csv": 
        rname = randomName() + ".csv"
        file_path = os.path.join(path, rname)
        print("THIS IS FILE PATH !!!!!!!!!",file_path)
        file.to_csv(file_path, index=False)

    elif ext in ["xls","xlsx","xlsm","xlsb"]:
        rname = randomName() + ".xlsx"
        file_path = os.path.join(path, rname)
        file.to_excel(file_path),
    else:
        pass
    return file_path


def extract_data(text):
    import re
    import io
    # Define the regex pattern to extract the CSV string
    pattern = r'<CSV>(.*?)<CSV>'
    match = re.search(pattern, text, re.DOTALL)

    if match:
        csv_string = match.group(1).strip()

        # Use io.StringIO to create a file-like object
        csv_file = io.StringIO(csv_string)

        # Read the CSV into a pandas DataFrame
        df = pd.read_csv(csv_file)
        try:
            os.mkdir("temp")
        except FileExistsError:
            pass
        n = randomName()+ ".csv"
        path = os.path.join('temp', n)
        df.to_csv(path, index=False)

        # Display the DataFrame
        print(df)
    else:
        print("No CSV string found in the text.")
    return path


def create_lida_data(file_path, file_name=None):
    import pandas as pd
    import os
    assert isinstance(file_path, list)
    try:
        os.mkdir('temp')
    except FileExistsError:
        pass
    file_name = f"LiDA{randomName()}"+".xlsx"
    lida_file_path = os.path.join('temp',file_name)
    with pd.ExcelWriter(lida_file_path) as writer:
        # Write each DataFrame to a different sheet
        for i,file in enumerate(file_path):
            sheet = f'Sheet{i}'
            df =  pd.read_csv(file)
            df.to_excel(writer, sheet_name=sheet, index=False)
            
    return lida_file_path
