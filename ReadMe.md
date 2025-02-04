AI-CPS_MS_PS

📚 Project Description

This repository is part of the course: "M. Grum: Advanced AI-based Application Systems"

Status: Finished

The target is to predict if the esitmated Earnings per Share (EPS) matches with the actual EPS from the
Earning History of the Apple Inc. (AAPL). This information helps the user to see if the estimated EPS can
be trusted or not.


👨‍💻 Installation

Prerequisites

Python 3.8 or higher

Git (optional, for cloning the repository)

Docker & Docker Compose (optional, for containerized deployment)

Clone the Repository

git clone https://github.com/Schummel5/AI-CPS_MS_PS.git
cd AI-CPS_MS_PS

Install Dependencies

pip install -r requirements.txt


🐳 Running with Docker Compose

To run the project using Docker Compose, use the following command:

docker compose -f docker-compose-ann.yml up
docker compose -f docker-compose-ols.yml up

This will start all necessary services in a containerized environment.


📜 Repository Structure

AI-CPS_MS_PS/
│-- code/               # Project source code
        |-- OLS_model/
        |-- data_scraping_preperation/
        |-- pyBrain/
│-- data/               # Sample or training data
        |-- ANN/
        |-- OLS_model
│-- documentation/      # Saved models or configuration files
│-- images/             # Docker images
│-- requirements.txt    # List of required Python packages
│-- docker-compose-ann.yml  # Docker Compose configuration
│-- docker-compose-ols.yml  # Docker Compose configuration
│-- README.md           # Documentation

👥 Contributors

This repository is maintained and owned by [Philip Schummel & Max Stavenhagen].  

💎 License

This project is released under the **AGPL-3.0 License**. You can view the full text of the license
[here](https://www.gnu.org/licenses/agpl-3.0.en.html).  

By utilizing this repository or its contents, you agree to comply with the terms of the AGPL-3.0 license, 
including its requirements for distribution and modification of derived works.  

If you have any questions or issues related to this project, feel free to contact the repository maintainers.  


