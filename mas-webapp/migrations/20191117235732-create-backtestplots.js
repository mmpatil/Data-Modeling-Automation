'use strict';

module.exports = {
  up: (queryInterface, Sequelize) => {
    return queryInterface.createTable('BackTestPlots', {
      Id: {
        allowNull: false,
        autoIncrement: true,
        primaryKey: true,
        type: Sequelize.INTEGER
      },
      ModelId: {
        type: Sequelize.INTEGER,
        references: {
          model: 'ModelRunDetail',
          key: 'Id'
        }
      },
      Name : Sequelize.STRING,
      JSON: Sequelize.TEXT,
      JSONType: Sequelize.TEXT
    });
  },

  down: (queryInterface, Sequelize) => {
    return queryInterface.dropTable('BackTestPlots');
  }
};
