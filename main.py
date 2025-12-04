from fastapi import FastAPI
import pandas as pd
import numpy as np
import pickle
from pydantic import BaseModel


class Input(BaseModel):
    job : str
    education : str
    avg_sal : int
    Years_Experience : float
    AI_Exposure_Index : float
    Tech_Growth_Factor: float
    Automation_Probability_2030:float




model = pickle.load(open("rf_model.pkl","rb"))


jobs = ['Security Guard', 'Research Scientist', 'Construction Worker',
       'Software Engineer', 'Financial Analyst', 'AI Engineer',
       'Mechanic', 'Teacher', 'HR Specialist', 'Customer Support',
       'UX Researcher', 'Lawyer', 'Data Scientist', 'Graphic Designer',
       'Retail Worker', 'Doctor', 'Truck Driver', 'Chef', 'Nurse',
       'Marketing Manager']

Education_Level = ["Master's", 'PhD', 'High School', "Bachelor's"]




app = FastAPI()


@app.get("/health")
def health_check():
    return {"Status : Healthy"}



@app.post("/input")
def input(input : Input):

    Years_Experience=input.Years_Experience
    job=input.job
    education=input.education
    avg_sal=input.avg_sal
    AI_Exposure_Index=input.AI_Exposure_Index
    Tech_Growth_Factor=input.Tech_Growth_Factor
    Automation_Probability_2030=input.Automation_Probability_2030

    input_data = pd.DataFrame([[job, education, avg_sal, Years_Experience, AI_Exposure_Index, Tech_Growth_Factor, Automation_Probability_2030]],
                              columns=['Job_Title', 'Education_Level', 'Average_Salary', 'Years_Experience', 'AI_Exposure_Index', 'Tech_Growth_Factor', 'Automation_Probability_2030'])
    
    prediction = model.predict(input_data)[0]

    return {"Prediction": str(prediction)}