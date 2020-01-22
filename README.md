## Overview  : Data Modeling Automation
Building appropriate financial time series models are essential to financial institutions as those models can be used to forecast the firm’s risk to changing economic environments. PNC bank, the sponsor of our project, has also been putting a significant amount of work to build the different types of models. While the data set is consistent across nearly every model, the modeling scripts written by different modelers differ drastically as each modeler has his or her own modeling methodologies. Therefore, there hasn’t been an easy, simple way for the team to run multiple modeling scripts other than running different modeling scripts each time. Also, the results from each modeling script are stored in various excel files, word documents and png images, which made it harder for the modelers to analyze those results. In short, the legacy process of running models was very time-consuming.
For the master project, USF-PNC team built a web application that effectively automates and simplifies the process of building different models of financial time series. It brute forces through various modeling forms, performs all the data transformation independently, and generates the top N models for consideration subject to user defined constraints. The output of selected models would include the statistical diagnostics of each model and graphical representation of dynamic backtesting and sensitivity testing. The saved outputs would also serve as audit trails for Federal Reserve audits of the financial institution as they will be legitimate evidence of the financial time series modeling procedures. The application significantly reduces the amount of redundancy and provides controls which reduce the likelihood of human error. Furthermore, it may be used to find candidate models more quickly and accurately, which could be very useful for users in benchmarking models against existing candidate models.

## Requirements
The project requirements were split into major sections: Data Science related requirements, Database related requirements, Web Application related requirements.
### Data Science Related Requirements
#### Core Functionalities
Writing data transformation script
Writing model output to the database
Saving data points for interactive graphs
Implementing One Factor Regression model
Extra Functionalities
Providing options to choose which statistical tests to be run
Refactoring and parameterizing the model script to make it reusable
### Database Related Requirements
#### Core Functionalities
Creating database schema based on the sponsors’ requests
Saving all the input values that user puts in when a job is started
Saving all the output values after each test
Creating database migration that standardizes the table creation and editing progress
Extra Functionalities
Creating additional tables to save data points of interactive graphs
Saving intermediate outputs after the preliminary statistical testing
### Web Application Related Requirements
#### Core Functionalities
Backend
Connecting the web application with database (MSSQL)
Connecting the web application with the data science python script
Handle saving and renaming of user uploaded files
Frontend
GET / - user can upload the dataset and configure tests
POST /jobs : handle form submission for new job by spawning the first child process for the preliminary data science script
GET /jobs - user can view the status of all jobs
GET /jobs/:id - user can view models that passed/failed the preliminary statistical tests and select them to be included in shortlist
POST /shortlist - handles form submission for selection of shortlist and spawn the script for additional testing such as backtesting, sensitivity tests, etc.
GET /model/:modelId - user can view detailed results of each model with graphs

### Extra Functionalities		
Spawning the second child process when the user select the models to be included in the shortlist
Each backtest date input can take more than one date
Interactive front end to add/remove fields
Made the graphs interactive with d3.js
