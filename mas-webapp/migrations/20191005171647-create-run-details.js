'use strict';

module.exports = {
  up: (queryInterface, Sequelize) => {
    return queryInterface.createTable('RunDetail', {
      Id: {
        allowNull: false,
        autoIncrement: true,
        primaryKey: true,
        type: Sequelize.INTEGER
      },
      StartDate: Sequelize.DATE,
      EndDate: Sequelize.DATE,
      Status: Sequelize.STRING,
      ModelType: Sequelize.STRING,
      EndDateTimeForTests: Sequelize.DATE
    });
  },

  down: (queryInterface, Sequelize) => {
    return queryInterface.dropTable('RunDetail');
  }
};
