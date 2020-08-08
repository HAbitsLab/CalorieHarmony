# CalorieHarmony

## Requirments
- joblib
- numpy
- pandas
- XGBoost (0.80)
- plotly
- pingouin
- sklearn
- scipy

## Data Location

Under selected data folder:

Place the csv that contains the participants' weight information here.
- name it "p_weights.csv" (Required column names: "Participant",  "Weight (kg)")

Then for each participant, create a separate folder using the participant name.

Under each participant folder:

Create in-lab and/or in-wild folder ("In Lab", "In Wild")

Under "In Lab" or "In Wild" folder:

Place the log csv file here. The log will contain the participant's timesheet information.

- Log file name: participant_name + in-lab/in-wild + log.csv, separated by space. (ex. "P431 In Lab Log.csv")

- in-lab csv columns:
    - "Activity" (activity name, ex. "computer")
    - "State" ("lab")
    - "Start Date" (activity start date, MM/DD/YY, ex. "3/10/20")
    - End Date (activity end date, MM/DD/YY, ex. "3/10/20")
    - Start Time (activity start time, HH/MM/SS, ex. "10:05:00")
    - End Time (activity end time, HH/MM/SS, ex. "10:10:00")
    - Start Calorie (whatever number indicated by Google Fit, ex. "600")
    - End Calorie  (whatever number indicated by Google Fit, ex. "700")
    - Trial Start (trail start time, "MM/DD/YY HH:MM", ex. "3/10/20 10:05")
    - Trial End (trail end time, "MM/DD/YY HH:MM", ex. "3/10/20 10:05")

- in-wild csv columns:
    - Same as in-lab but replace "Activity" with "Day", (free-living day number, ex. "1")

Make sure the wrist data are pre-processed. The wrist data files (accelerometer and gyroscope) need to be one csv file per participant.

Accelerometer data
- rename to accelerometer.csv and put under /wrist/accelerometer
- column names: Time, accX, accY, accZ (Time needs to be in 12 digits unix timestamp)

Gyroscope data
- rename to gyroscope.csv and put under /wrist/gyroscope
- column names: Time, rotX, rotY, rotZ (Time needs to be in 12 digits unix timestamp)

ActiGraph output file
- rename to actigraph.csv and put under /actigraph

## Run Order

In terminal, cd to the repository directory.
 
Then run in below order:

> preprocressing.py:  

In terminal: type "python preprocressing.py" followed by the path to data folder, participant names separated by space, then "In Lab" or "In Wild".

- ex. python preprocressing.py "my/path/to/data/folder" "P301 P302 P401" "In Lab"

This will use the wrist and ActiGraph data to generate minute level data needed for building the model.

> cv.py

In terminal: type "python cv.py" followed by the path to data folder, then participant names separated by space.

- ex. python3 cv.py "my/path/to/data/folder" "P301 P302 P401"

This will use the in-lab data of these participants to build and evaluate the WRIST model using cross validation. It also outputs some result graphs

To regenerate the paper result, use "P401 P404 P405 P409 P410 P411 P415 P417 P418 P421 P422 P423 P424 P425 P427 P429 P431" from the provided sample data set.

> compare_wild.py

In terminal: type "python compare_wild.py" followed by the path to data folder, then participant names separated by space.

- ex. python3 compare_wild.py "my/path/to/data/folder" "P301 P302 P401"

This will use the in-wild data of these participants to estimate and compare the results of WRIST and ActiGraph.

To regenerate the paper result, use "P404 P405 P409 P410 P411 P412 P415 P416 P417 P418 P419 P420 P421 P423 P424 P425 P427 P429 P431"
 from the provided sample data set.

> stat_for_paper.py

In terminal: "python stat_for_paper.py"

Generate all the numbers used in the paper.

## DOCKER BUILD

Code to build the docker image
The docker file is included to build. 
A public image is available on docker hub to run without the need to build locally

In order to view all of the output and running enter the interactive terminal (*-it*)

Download and unzip the output_files.zip locally

The docker container needs to share data with the host system (*-v*) the user might need to configure docker to share the local file system with docker

file structure of sample data on host system:
```
output_files
├── sampleData
│   └── P401
│   └── P404
│   └── P405
│   └── ...
│   └── p_weights.csv
```

> docker run -it -v [LOCAL DIRECTORY WITH output_files]:/output_files habitslab/calorie-harmony bash

Example:
>  docker run -it -v  C:\Users\tpt4349\Documents\CalorieHarmony\output_files:/output_files habitslab/calorie-harmony bash

This places you in the terminal for the docker container then you can run the **./run.sh** command to start the WRIST algorithm
> root@3f6523f1e595:/# ./run.sh

It will take a few hours to preprocess the data, create to model and output all of the stats.

All of the outputs will be saved on the host system under the specifier root output_files directory.

The target outputs from the paper will saved in a file named **stat_for_paper.txt** under the output_files directory on the host system

If the user runs the docker image without the *-it* flag it will run silently in background and save the output data with not views or interaction from user.

Example:
>  docker run -v  C:\Users\tpt4349\Documents\CalorieHarmony\output_files:/output_files habitslab/calorie-harmony
