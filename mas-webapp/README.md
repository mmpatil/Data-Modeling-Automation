# Webapp
## Setup
https://nodejs.org/en/

- make sure you have node js and npm installed
- (Mac Specific) make sure you have docker and azure data studio installed (https://database.guide/how-to-install-sql-server-on-a-mac/)
	- for our needs you only need to complete steps 1-5 in the linked guide
- create database(s) for the webapp (mas_development, mas_test and mas_production)
- create a config.json located in mas-webapp/config/config.json
	- there is an exampleconfig.json in that folder
	- you can copy the contents of exampleconfig.json to a new file named config.json
	- at minimum replace the username and password in the example with the username and password from starting the docker instance
- python side needs to install
```
pip3 install Cython
brew install freetds
pip3 install pymssql
pip3 install sqlalchemy
```

### Migrations
(this creates the tables in the database)
Prerequisite installations
```
npm install -g mssql
npm install -g tedious
npm install -g sequelize
npm install -g sequelize-cli

```
To run
- in the mas-webapp directory
```
sequelize db:migrate
```

### Seeding (optional)
populate database with some sample data.  You should only run this once and just if you need to fill an empty database with some test/sample data
```
sequelize-cli db:seed:all
```
## How to Run Webapp
- cd into mas-webapp directory
- run the following the first time and every time changes are pulled
(may need to run npm update if you have a previous installation and haven't updated in a while)
```
npm install
```
- to start normally
```
npm start
```
- to start in dev and debug mode
```
DEBUG=mas-webapp:* npm run devstart
```
