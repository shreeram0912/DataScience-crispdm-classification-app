## **Acknowledgement:**
* I would like to express my heartfelt gratitude to my mentor, B Sathiyabama, for her invaluable guidance, support, and encouragement throughout the development of the CRISP-DM Classification project. Her mentorship has
  been instrumental in shaping this work.
* I also extend my sincere thanks to Gustavo R Santos, a talented data scientist, whose insightful online tutorials and resources played a crucial role in deepening my understanding and aiding the successful completion of this project.
* I would also like to thank my family and friends for their constant support and encouragement throughout the project journey.
* Thank you all for being a part of this remarkable learning experience.


# DataScience-crispdm-classification-app
The project focus on understanding CRISP-DM (Cross-Industry Standard Process for Data Mining) is a widely used framework for data mining and analytics projects. It provides a structured, step-by-step approach to handling data science tasks efficiently.

![image](https://github.com/user-attachments/assets/902a43eb-a309-42a1-aa61-60a1d5cdefa0)

## **Introduction**
CRISP-DM stands for Cross-Industry Standard Process for Data Mining, a data mining framework open to anyone who wants to use it.
* Its first version was created by SPSS, Daimler-Benz and NCR. Then, a group of companies developed and evolved it to CRISP-DM, which nowadays is one of the most known and adopted frameworks in Data Science.
* The process consists of 6 phases, and it is flexible. It is more like a living organism where you can (and probably should) go back and forth between the phases, iterating and enhancing the results.
* The phases are:
  + Business Understanding – Define project objectives and success criteria.
  + Data Understanding – Collect, explore, and analyze data to identify patterns.
  + Data Preparation – Clean, transform, and structure data for analysis.
  + Modeling – Apply machine learning or statistical techniques to build predictive models.
  + Evaluation – Assess model performance and ensure it meets business goals.
  + Deployment – Implement the model in a real-world setting for decision-making.
  Iterative Nature: At any stage, you might loop back based on findings (e.g., poor model → revisit data prep).
* The small arrows show a natural path from Business Understanding to Deployment—where the interactions occur directly—while the circle denotes a cyclic relationship between the phases. This means that the project does not end
  with Deployment but can be restarted due to new business questions triggered by the project or adjustments potentially needed.
* For Business understanding, Data understanding, Data Preparation, Modeling & evaluation refer to: https://github.com/shreeram0912/DataScience-crispdm-classification-app/blob/main/Project_Bank.ipynb
* For Deployment: Coming soon.
  
![image](https://github.com/user-attachments/assets/911f97b7-627f-4cea-84f4-c7b88320640d)


## **Business Problem**
* Business Problem: Our customer, a Bank Institution, wants to optimize sales and their manager's time when selling financial products. The product for this project is a Term Direct Deposit.
* A fixed-term investment where money is deposited into an account at a financial institution. Term deposits are also known as certificates of deposit (CDs) or time deposits. Term deposits typically have higher interest rates
  than traditional savings accounts, but the funds are not accessible until the term ends.
* Proposed Solution: This project aims to create classifier to predict the probability of a customer to convert when offered a financial product (direct term deposit) via a phone call.


## **Dataset:**
* The Dataset can be found in UCI DS Repository Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing [Dataset]. UCI Machine Learning Repository.


## **Overview**
The application takes several customer-related features as input and uses a trained machine learning model to predict the probability of the customer subscribing to a term deposit. The predicted probability is then displayed along with a visual representation.


## **Features**
* User-Friendly Interface: Built with Streamlit, providing an intuitive web interface for inputting customer data.
* Real-time Prediction: Predicts the conversion probability based on the entered features.
* Visual Representation: Displays the predicted probability using a bar chart for easy interpretation.
* Clear Output: Shows the probability as a percentage, making it easy to understand.


## **Prerequisites**
Before running the application, ensure you have the following installed:
* Python 3.6 or higher
* Required Python Libraries:
1. Pandas 🐼: Used for data manipulation and analysis, Provides powerful data structures like DataFrames and Series.
2. NumPy 🔢: Fundamental package for numerical computing, Enables efficient handling of large arrays and matrices.
              Supports mathematical operations like linear algebra, statistics, and Fourier transforms.
3. Matplotlib 📊: Used for creating static, animated, and interactive visualizations, Provides functions to plot graphs, histograms, bar charts, and scatter plots.
4. Seaborn 🎨: Built on top of Matplotlib, designed for statistical data visualization, Works seamlessly with Pandas DataFrames.
5. Scikit-learn 🤖: A machine learning library with tools for classification, regression, clustering, and dimensionality reduction, Includes algorithms like decision trees, support 
                     vector machines, and random forests, Provides utilities for model evaluation and preprocessing.
6. CatBoost 🏆: A gradient boosting library optimized for categorical data, Used for high-performance machine learning tasks like ranking, classification, and regression, Requires 
                 minimal preprocessing and handles missing values efficiently.
7. Feature-engine 🔍: A library for feature engineering in machine learning, Helps with transformations like missing value imputation, encoding categorical variables, and scaling 
                       numerical features, Works well with Scikit-learn pipelines.
8. category-encoders 🏷️: Encode categorical variables into numeric formats compatible with scikit-learn pipelines.
9. ipykernel ⚡: Enables interactive Python sessions through the IPython kernel for Jupyter.
10. mlflow 🚀: Manages the machine learning lifecycle, covering experimentation, reproducibility, and deployment.
11. nbformat 📚: Handles the structure and format of Jupyter Notebook files.
12. plotly 📊: Creates interactive, high-quality visualizations for Python.
13. streamlit 🌐: Builds interactive web apps for machine learning and data science.
14. ucimlrepo 📂: A repository offering diverse datasets for machine learning research.
15. pip 🛠️: A package manager for Python that simplifies the installation, upgrade, and management of Python libraries.
16. UV 🚀: A modern, high-performance Python package manager written in Rust, offering faster dependency management and virtual environment creation
* Docker 🐳: A platform for developing, shipping, and running applications in lightweight, portable containers.
* Jupyter Notebook 📓: An interactive environment for writing and running live code, visualizing data, and combining narrative text with code execution. Perfect for data analysis and exploration!


## * **How to use App in your localhost:**
1. Clone the repository: Run git clone https://github.com/shreeram0912/DataScience-crispdm-classification-app.git in your terminal. This will download the CRISP-DM repository to your local machine.
2. Before starting building process process make sure to install docker
3. Start the application using Docker Compose: Navigate to the crispd directory: cd crispdm
4. Run docker-compose up --build to build and start the containers.
5. Start a container from the image example: docker run -p 8501:8501 -p 5000:5000 crispdm-classification-app

## How to use Mlflow, Docker in Visual Studio Code:
*   **Mlflow**
1. pip install mlflow or uv add mlflow
2. mlflow server, by default it run on localhost:5000
*   **Docker**
1. Install Docker on Machine
2. Create docker file in visual studio code, before creating docker image make sure docker is running
3. Create an image example: docker build -t crispdm-classification-app .
4. Start a container from the image example: docker run -p 8501:8501 -p 5000:5000 crispdm-classification-app
5. Stop the container: docker stop <container-id>
6. List of Docker images: docker images
7. Check whether container is running: docker ps
## *  **Github: version control system**
1. git init
2. git add .
3. git commit -m "Initial commit for my Streamlit app"
4. git remote add origin https://github.com/shreeram0912/DataScience-crispdm-classification-app.git
5. git branch -M main
6. git push -u origin main


**Docker:**
Docker Image running: ![image](https://github.com/user-attachments/assets/69ca2fbc-13cf-4ea6-9cdb-0bcc64780a37)
**Docker App: Build history**
![image](https://github.com/user-attachments/assets/d518196c-807c-4b95-853d-e58f13303c4e)
![image](https://github.com/user-attachments/assets/6df6ec77-daa7-4e41-8392-e8a0827a712a)
![image](https://github.com/user-attachments/assets/22d77ead-86ec-4071-bc80-2f69e5ecd4d1)
Mlflow:
![image](https://github.com/user-attachments/assets/bfaeae29-70e1-4a30-937c-0db86f006395)
![image](https://github.com/user-attachments/assets/db52dec6-d976-4c87-b4e4-1ba86428d709)


## **Model**
* The application relies on a pre-trained classification model (saved as model6.pkl). This model has been trained to predict the likelihood of a customer subscribing to a term deposit based on the provided features.
* Note: The accuracy and performance of the application are directly dependent on the quality and performance of the underlying machine learning model.
Streamlit App:
![image](https://github.com/user-attachments/assets/56fb0b74-6eee-4a7d-9c31-399bfe1e0352)
![image](https://github.com/user-attachments/assets/f27bb2d5-68e3-4643-8614-c89a77b6082e)
App Demonstration:
https://drive.google.com/file/d/1YB_bHCXSGh2LW1cBBh2OB8yOcI0IvuYk/view?usp=sharing


## **References:** 
https://en.wikipedia/wiki/Cross-industry standard process for data mining
https://www.datascience-pm.com/crisp-dm-2/
https://www.ibm.com/docs/sr/spss-modeler/saas?topic=dm-crisp-help-overview
https://tinyurl.com/crispdm-bank-prj-desc


## **Contact:**
sprajapati99.sp@gmail.com
www.linkedin.com/in/shreeram-prajapati-11255631b
